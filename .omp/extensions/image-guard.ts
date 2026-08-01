import { spawn, type ChildProcessByStdio } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import type { Readable } from "node:stream";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

const PROTOCOL_VERSION = 1;
const READY_TIMEOUT_MS = 60_000;
const MAX_PROTOCOL_LINE_BYTES = 1_048_576;
const STATUS_KEY = "image-guard";
const STATE_MESSAGE_TYPE = "image-guard-state";

type GuardSnapshot =
	| { version: 1; state: "clear"; paths: [] }
	| { version: 1; state: "blocked"; paths: string[] };
type WatcherState =
	| { kind: "starting" }
	| { kind: "clear" }
	| { kind: "blocked"; paths: string[] }
	| { kind: "restarting"; reason: string }
	| { kind: "failed"; reason: string };
type GuardChild = ChildProcessByStdio<null, Readable, Readable>;

class ImageGuardController {
	private child: GuardChild | undefined;
	private state: WatcherState = { kind: "starting" };
	private knownPaths: string[] = [];
	private readinessTimer: NodeJS.Timeout | undefined;
	private settleStartup: (() => void) | undefined;
	private automaticRestartUsed = false;
	private shuttingDown = false;
	private initialized = false;

	constructor(private readonly pi: ExtensionAPI) {}

	register(): void {
		this.pi.on("session_start", async (_event, ctx) => this.start(ctx));
		this.pi.on("context", (event) => {
			const nag = this.contextNag();
			if (!nag) return;
			const message: AgentMessage = {
				role: "custom",
				customType: STATE_MESSAGE_TYPE,
				content: nag,
				display: false,
				attribution: "user",
				timestamp: Date.now(),
			};
			return { messages: [...event.messages, message] };
		});
		this.pi.on("session_shutdown", async (_event, ctx) => this.shutdown(ctx));
		this.registerRecoveryTool();
	}

	private registerRecoveryTool(): void {
		this.pi.registerTool({
			name: "image_guard_retry",
			label: "Retry Image Guard",
			description: "Retry the failed repository image watcher after repairing its dependency or protocol failure.",
			parameters: this.pi.zod.object({}),
			execute: async (_id, _params, _signal, _update, ctx) => {
				if (this.state.kind !== "failed") {
					const text = `Image guard is ${this.state.kind}; retry is only available after failure.`;
					return { content: [{ type: "text" as const, text }], details: { state: this.state.kind }, isError: false };
				}
				this.automaticRestartUsed = false;
				this.state = { kind: "starting" };
				ctx.ui.setStatus(STATUS_KEY, "Image guard: starting recovery");
				await this.beginStartup(ctx);
				const failed = this.state.kind === "failed";
				const text = failed ? `Image guard recovery failed: ${this.state.reason}` : "Image guard recovered.";
				return { content: [{ type: "text" as const, text }], details: { state: this.state.kind }, isError: failed };
			},
		});
	}

	private async start(ctx: ExtensionContext): Promise<void> {
		if (this.initialized) return;
		this.initialized = true;
		this.shuttingDown = false;
		this.state = { kind: "starting" };
		ctx.ui.setStatus(STATUS_KEY, "Image guard: starting");
		await this.beginStartup(ctx);
	}

	private async beginStartup(ctx: ExtensionContext): Promise<void> {
		await new Promise<void>((resolve) => {
			this.settleStartup = resolve;
			this.spawnWatcher(ctx);
		});
	}

	private spawnWatcher(ctx: ExtensionContext): void {
		let child: GuardChild;
		try {
			child = spawn("task", ["image-guard"], {
				cwd: ctx.cwd,
				detached: true,
				stdio: ["ignore", "pipe", "pipe"],
			});
		} catch (error: unknown) {
			void this.handleFailure(`could not spawn task image-guard: ${formatError(error)}`, ctx);
			return;
		}
		this.child = child;
		this.observeProcess(child, ctx);
		this.clearReadinessTimer();
		this.readinessTimer = setTimeout(() => {
			void this.failCurrent(child, `readiness was not received within ${READY_TIMEOUT_MS}ms`, ctx);
		}, READY_TIMEOUT_MS);
	}

	private observeProcess(child: GuardChild, ctx: ExtensionContext): void {
		child.stdout.on("data", this.protocolReader(child, ctx));
		child.stderr.on("data", (chunk: Buffer) => {
			this.pi.logger.debug(`image-guard watcher: ${chunk.toString().trimEnd()}`);
		});
		child.once("error", (error: Error) => {
			void this.failCurrent(child, `task image-guard process error: ${error.message}`, ctx);
		});
		child.once("close", (code: number | null, signal: NodeJS.Signals | null) => {
			if (this.shuttingDown || child !== this.child) return;
			const outcome = signal ? `signal ${signal}` : `exit code ${String(code)}`;
			void this.failCurrent(child, `task image-guard exited unexpectedly (${outcome})`, ctx);
		});
	}

	private protocolReader(child: GuardChild, ctx: ExtensionContext): (chunk: Buffer) => void {
		const decoder = new StringDecoder("utf8");
		let pending = "";
		return (chunk: Buffer): void => {
			if (child !== this.child) return;
			pending += decoder.write(chunk);
			if (Buffer.byteLength(pending) > MAX_PROTOCOL_LINE_BYTES) {
				void this.failCurrent(child, "watcher protocol line exceeded the size limit", ctx);
				return;
			}
			const lines = pending.split("\n");
			pending = lines.pop() ?? "";
			for (const line of lines) {
				if (child !== this.child) return;
				this.inspectSnapshotLine(line, child, ctx);
			}
		};
	}

	private inspectSnapshotLine(line: string, child: GuardChild, ctx: ExtensionContext): void {
		let value: unknown;
		try {
			value = JSON.parse(line);
		} catch (error: unknown) {
			void this.failCurrent(child, `invalid watcher JSON: ${formatError(error)}`, ctx);
			return;
		}
		const snapshot = validateSnapshot(value);
		if (!snapshot) {
			void this.failCurrent(child, "invalid watcher snapshot", ctx);
			return;
		}
		this.clearReadinessTimer();
		this.applySnapshot(snapshot, ctx);
		this.finishStartup();
	}

	private applySnapshot(snapshot: GuardSnapshot, ctx: ExtensionContext): void {
		const previousPaths = this.knownPaths;
		this.knownPaths = [...snapshot.paths];
		if (snapshot.state === "clear") {
			this.state = { kind: "clear" };
			ctx.ui.setStatus(STATUS_KEY, undefined);
			if (previousPaths.length > 0) this.sendResume();
			return;
		}
		this.state = { kind: "blocked", paths: [...snapshot.paths] };
		ctx.ui.setStatus(STATUS_KEY, blockedStatus(snapshot.paths));
		const previous = new Set(previousPaths);
		for (const path of snapshot.paths) {
			if (!previous.has(path)) this.sendPathSteer(path);
		}
	}

	private sendPathSteer(path: string): void {
		this.pi.sendMessage(
			{
				customType: STATE_MESSAGE_TYPE,
				content: `Image guard detected ${path}. Never inspect, read, or display this path. Cleanup is the top priority unless Scott explicitly directs otherwise. Tools remain unrestricted for cleanup.`,
				display: false,
				attribution: "user",
			},
			{ deliverAs: "steer", triggerTurn: true },
		);
	}

	private sendResume(): void {
		this.pi.sendMessage(
			{
				customType: STATE_MESSAGE_TYPE,
				content: "Image guard is clear. Resume the task that was interrupted for cleanup.",
				display: false,
				attribution: "user",
			},
			{ deliverAs: "steer", triggerTurn: true },
		);
	}

	private contextNag(): string | undefined {
		if (this.state.kind === "blocked") {
			return `IMAGE GUARD BLOCKED PATHS: ${this.state.paths.join(", ")}. Never inspect, read, or display these paths. Cleanup is the top priority unless Scott explicitly directs otherwise. Tools remain unrestricted for cleanup.`;
		}
		if (this.state.kind === "failed") {
			return `IMAGE GUARD FAILED: ${this.state.reason}. Restore the watcher as the top priority, then call image_guard_retry. Never inspect, read, or display any previously reported path. Tools remain unrestricted for repair.`;
		}
		return undefined;
	}

	private async failCurrent(child: GuardChild, reason: string, ctx: ExtensionContext): Promise<void> {
		if (this.shuttingDown || child !== this.child) return;
		this.child = undefined;
		this.clearReadinessTimer();
		await this.terminateChild(child);
		await this.handleFailure(reason, ctx);
	}

	private async handleFailure(rawReason: string, ctx: ExtensionContext): Promise<void> {
		if (this.shuttingDown) return;
		const reason = sanitizeReason(rawReason);
		if (!this.automaticRestartUsed) {
			this.automaticRestartUsed = true;
			this.state = { kind: "restarting", reason };
			ctx.ui.setStatus(STATUS_KEY, `Image guard: restarting after ${reason}`);
			this.spawnWatcher(ctx);
			return;
		}
		this.state = { kind: "failed", reason };
		ctx.ui.setStatus(STATUS_KEY, `IMAGE GUARD FAILED: ${reason}`);
		this.sendFailureSteer(reason);
		this.finishStartup();
	}

	private sendFailureSteer(reason: string): void {
		this.pi.sendMessage(
			{
				customType: STATE_MESSAGE_TYPE,
				content: `Image guard failed: ${reason}. Restore the watcher as the top priority, then call image_guard_retry. Never inspect, read, or display any previously reported path. Tools remain unrestricted for repair.`,
				display: false,
				attribution: "user",
			},
			{ deliverAs: "steer", triggerTurn: true },
		);
	}

	private clearReadinessTimer(): void {
		if (!this.readinessTimer) return;
		clearTimeout(this.readinessTimer);
		this.readinessTimer = undefined;
	}

	private finishStartup(): void {
		if (!this.settleStartup) return;
		this.settleStartup();
		this.settleStartup = undefined;
	}

	private signalProcessGroup(child: GuardChild, signal: NodeJS.Signals): void {
		if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
		try {
			process.kill(-child.pid, signal);
		} catch (error: unknown) {
			if ((error as NodeJS.ErrnoException).code !== "ESRCH") {
				this.pi.logger.warn(`image-guard: failed to send ${signal}: ${formatError(error)}`);
			}
		}
	}

	private async terminateChild(child: GuardChild): Promise<void> {
		if (child.exitCode !== null || child.signalCode !== null) return;
		this.signalProcessGroup(child, "SIGTERM");
		const exited = await new Promise<boolean>((resolve) => {
			const timer = setTimeout(() => resolve(false), 500);
			child.once("exit", () => {
				clearTimeout(timer);
				resolve(true);
			});
		});
		if (!exited) this.signalProcessGroup(child, "SIGKILL");
	}

	private async shutdown(ctx: ExtensionContext): Promise<void> {
		this.shuttingDown = true;
		this.clearReadinessTimer();
		this.finishStartup();
		const child = this.child;
		this.child = undefined;
		if (child) await this.terminateChild(child);
		ctx.ui.setStatus(STATUS_KEY, undefined);
	}
}

function validateSnapshot(value: unknown): GuardSnapshot | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const record = value as Record<string, unknown>;
	if (record.version !== PROTOCOL_VERSION || !Array.isArray(record.paths)) return undefined;
	if (Object.keys(record).sort().join(",") !== "paths,state,version") return undefined;
	if (!record.paths.every(isSafePath)) return undefined;
	const paths = record.paths as string[];
	if (paths.some((path, index) => index > 0 && paths[index - 1]! >= path)) return undefined;
	if (record.state === "clear" && paths.length === 0) return { version: 1, state: "clear", paths: [] };
	if (record.state === "blocked" && paths.length > 0) return { version: 1, state: "blocked", paths };
	return undefined;
}

function isSafePath(value: unknown): value is string {
	return typeof value === "string"
		&& value.length > 0
		&& value !== "."
		&& !value.startsWith("/")
		&& !value.split("/").includes("..");
}

function blockedStatus(paths: readonly string[]): string {
	const shown = paths.slice(0, 2).map((path) => path.replace(/[\u0000-\u001f\u007f]/g, "?"));
	const suffix = paths.length > shown.length ? `, +${paths.length - shown.length} more` : "";
	return `IMAGE GUARD: ${paths.length} path(s): ${shown.join(", ")}${suffix}`;
}

function sanitizeReason(reason: string): string {
	return reason.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, 240) || "unknown failure";
}

function formatError(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

export default function imageGuardExtension(pi: ExtensionAPI): void {
	new ImageGuardController(pi).register();
}
