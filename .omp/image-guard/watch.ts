import { spawn, type ChildProcessByStdio } from "node:child_process";
import { readdir } from "node:fs/promises";
import type { Dirent } from "node:fs";
import { join, relative, sep } from "node:path";
import type { Readable } from "node:stream";

const ROOT = ".";
const PROTOCOL_VERSION = 1 as const;

//! Vendor and generated directories are excluded from the recursive scan.
//! The guard exists to catch spider imagery that
//! enters *project source* — files Scott and the agents author — not to police
//! third-party packages. Installed dependencies ship legitimate image assets
//! (favicons, example PNGs, SVG diagrams), and Python wheels embed ONNX
//! protobuf tensor fixtures whose raw bytes incidentally satisfy `file`'s
//! image magic-byte sniff, so `file --mime-type` reports them as `image/*`.
//! Walking these trees floods the guard with false positives that drown real
//! detections. Add a name here only for directories that are regenerated,
//! third-party, or otherwise outside human-authored source.
const EXCLUDED_DIRS = new Set<string>([
	"node_modules", // JavaScript dependencies (pnpm).
	".venv", "venv", // Python virtual environments (uv).
	".git", // Version-control metadata.
	".output", ".wxt", "dist", "build", "target", // Generated build artifacts.
	"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".basedpyright", // Python tool caches.
]);

type GuardSnapshot =
	| { version: typeof PROTOCOL_VERSION; state: "clear"; paths: [] }
	| { version: typeof PROTOCOL_VERSION; state: "blocked"; paths: string[] };
type MonitorChild = ChildProcessByStdio<null, Readable, Readable>;

class ImageWatcher {
	private monitor: MonitorChild | undefined;
	private previousPaths: string[] | undefined;
	private scanning = false;
	private scanAgain = false;
	private refreshingMonitor = false;
	private eventPending = "";

	run(): void {
		this.launchMonitor(true);
	}

	private launchMonitor(initial: boolean): void {
		const child = spawn(
			"inotifywait",
			[
				"--monitor", "--recursive",
				"--event", "close_write,moved_to,delete,moved_from,create",
				"--format", "%e", ROOT,
			],
			{ env: { ...process.env, LC_ALL: "C" }, stdio: ["ignore", "pipe", "pipe"] },
		);
		let activated = false;
		let stderrPending = "";
		child.stdout.on("data", (chunk: Buffer) => this.inspectEvents(child, chunk));
		child.stderr.on("data", (chunk: Buffer) => {
			stderrPending += chunk.toString();
			const lines = stderrPending.split("\n");
			stderrPending = lines.pop() ?? "";
			for (const line of lines) {
				if (!activated && line === "Watches established.") {
					activated = true;
					this.activateMonitor(child, initial);
				} else if (activated && line) process.stderr.write(`image-guard: ${line}\n`);
			}
		});
		child.once("error", (error: Error) => this.abort(`inotifywait error: ${error.message}`));
		child.once("close", (code: number | null, signal: NodeJS.Signals | null) => {
			if (activated && child !== this.monitor) return;
			const outcome = signal ? `signal ${signal}` : `exit code ${String(code)}`;
			this.abort(`inotifywait stopped unexpectedly (${outcome})`);
		});
	}

	private activateMonitor(child: MonitorChild, initial: boolean): void {
		const previous = this.monitor;
		this.monitor = child;
		this.eventPending = "";
		this.refreshingMonitor = false;
		if (previous && previous !== child) previous.kill("SIGTERM");
		if (initial || previous) this.requestScan();
	}

	private inspectEvents(child: MonitorChild, chunk: Buffer): void {
		if (child !== this.monitor) return;
		this.eventPending += chunk.toString();
		const lines = this.eventPending.split("\n");
		this.eventPending = lines.pop() ?? "";
		const events = lines.filter(Boolean);
		if (events.length === 0) return;
		this.requestScan();
		if (!this.refreshingMonitor && events.some((event) => event.includes("ISDIR"))) {
			this.refreshingMonitor = true;
			this.launchMonitor(false);
		}
	}

	private requestScan(): void {
		if (this.scanning) {
			this.scanAgain = true;
			return;
		}
		this.scanning = true;
		void this.scanUntilSettled();
	}

	private async scanUntilSettled(): Promise<void> {
		try {
			do {
				this.scanAgain = false;
				this.publish(await scanImages(ROOT));
			} while (this.scanAgain);
		} catch (error: unknown) {
			const reason = error instanceof Error ? error.message : String(error);
			this.abort(`scan failed: ${reason}`);
		} finally {
			this.scanning = false;
		}
	}

	private publish(paths: string[]): void {
		const unchanged = this.previousPaths
			&& this.previousPaths.length === paths.length
			&& this.previousPaths.every((value, index) => value === paths[index]);
		if (unchanged) return;
		this.previousPaths = paths;
		const snapshot: GuardSnapshot = paths.length === 0
			? { version: PROTOCOL_VERSION, state: "clear", paths: [] }
			: { version: PROTOCOL_VERSION, state: "blocked", paths };
		process.stdout.write(`${JSON.stringify(snapshot)}\n`);
		if (paths.length > 0) {
			process.stderr.write(`\aIMAGE GUARD: ${paths.length} image path(s) require cleanup. Watcher continues.\n`);
		}
	}

	private abort(reason: string): never {
		process.stderr.write(`image-guard: ${reason}\n`);
		process.exit(1);
	}
}

async function scanImages(directory: string): Promise<string[]> {
	const candidates: string[] = [];
	await collectCandidates(directory, candidates);
	const images: string[] = [];
	for (let offset = 0; offset < candidates.length; offset += 200) {
		const batch = candidates.slice(offset, offset + 200);
		const mimeTypes = await classifyBatch(batch);
		for (let index = 0; index < batch.length; index += 1) {
			if (mimeTypes[index]?.startsWith("image/")) images.push(displayPath(batch[index]!));
		}
	}
	return images.sort();
}

async function collectCandidates(directory: string, candidates: string[]): Promise<void> {
	let entries: Dirent[];
	try {
		entries = await readdir(directory, { withFileTypes: true });
	} catch (error: unknown) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
		throw error;
	}
	for (const entry of entries) {
		if (entry.isDirectory() && EXCLUDED_DIRS.has(entry.name)) continue;
		const path = join(directory, entry.name);
		if (entry.isDirectory()) await collectCandidates(path, candidates);
		else if (entry.isFile() || entry.isSymbolicLink()) candidates.push(path);
	}
}

async function classifyBatch(paths: string[]): Promise<string[]> {
	return await new Promise<string[]>((resolve, reject) => {
		const child = spawn("file", ["--brief", "--dereference", "--mime-type", "--", ...paths], {
			stdio: ["ignore", "pipe", "ignore"],
		});
		let output = "";
		child.stdout.on("data", (chunk: Buffer) => { output += chunk.toString(); });
		child.once("error", reject);
		child.once("close", (code: number | null) => {
			if (code !== 0) {
				reject(new Error(`file exited with code ${String(code)}`));
				return;
			}
			const mimeTypes = output.trimEnd().split("\n");
			if (mimeTypes.length === paths.length) resolve(mimeTypes);
			else reject(new Error(`file returned ${mimeTypes.length} results for ${paths.length} paths`));
		});
	});
}

function displayPath(path: string): string {
	return relative(ROOT, path).split(sep).join("/");
}

new ImageWatcher().run();
