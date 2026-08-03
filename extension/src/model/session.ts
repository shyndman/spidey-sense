import {
  InferenceSession,
  type InferenceSession as OnnxInferenceSession,
} from "onnxruntime-web/wasm";

import type { InferenceSessionLike } from "./inference";
import {
  loadModelMetadata,
  type ModelMetadata,
  resolveModelUrl,
} from "./metadata";
import { initializeOnnxRuntime } from "./runtime-environment";

const WASM_EXECUTION_PROVIDERS = ["wasm"] as const;

export type ModelSessionState =
  | "idle"
  | "initializing"
  | "ready"
  | "failed";
export type ModelGraphBoundary = "input" | "output";

/** A graph mismatch detected before any caller can perform inference. */
export class ModelGraphContractError extends Error {
  readonly code = "GRAPH_MISMATCH";
  readonly boundary: ModelGraphBoundary;

  constructor(boundary: ModelGraphBoundary) {
    super(`ONNX session ${boundary} graph does not match generated metadata`);
    this.name = "ModelGraphContractError";
    this.boundary = boundary;
  }
}

/**
 * The runtime surface owned by model initialization. It contains the graph
 * metadata required for contract validation, the inference method consumed by
 * `runModelInference`, and explicit resource release for rejected sessions.
 */
export type ModelRuntimeSession = InferenceSessionLike &
  Pick<
    OnnxInferenceSession,
    | "inputNames"
    | "outputNames"
    | "inputMetadata"
    | "outputMetadata"
    | "release"
  >;

export interface ReadyModelSession {
  readonly metadata: ModelMetadata;
  readonly session: ModelRuntimeSession;
}

export interface ModelSessionDependencies {
  readonly loadMetadata: (metadataUrl: URL) => Promise<ModelMetadata>;
  readonly createSession: (modelUrl: URL) => Promise<ModelRuntimeSession>;
}

interface ExpectedTensorBoundary {
  readonly name: string;
  readonly dataType: "float32";
  readonly shape: readonly (number | null)[];
}

const defaultDependencies: ModelSessionDependencies = {
  loadMetadata: loadModelMetadata,
  createSession: async (modelUrl) =>
    initializeOnnxRuntime(async () =>
      InferenceSession.create(modelUrl.href, {
        executionProviders: WASM_EXECUTION_PROVIDERS,
      }),
    ),
};

/**
 * Owns the extension's single initialization attempt for one packaged model.
 * Metadata is validated before its relative model artifact is loaded; the
 * resulting ONNX session is exposed only after its graph matches that metadata.
 * Every caller receives the same promise and ready session, including callers
 * arriving while initialization is in progress, so concurrent feature paths
 * cannot create duplicate WASM sessions.
 *
 * Failure is sticky for the lifetime of this manager: later callers receive the
 * same rejected promise rather than silently retrying an expensive or invalid
 * model load. An explicit new manager instance is required for a new attempt.
 * Logs contain only stable stage text—never URLs, model bytes, tensors, or image
 * data—while the original typed error remains available to the caller.
 */
export class ModelSessionManager {
  readonly #metadataUrl: URL;
  readonly #dependencies: ModelSessionDependencies;
  #state: ModelSessionState = "idle";
  #initialization: Promise<ReadyModelSession> | undefined;

  constructor(
    metadataUrl: URL,
    dependencies: ModelSessionDependencies = defaultDependencies,
  ) {
    this.#metadataUrl = new URL(metadataUrl.href);
    this.#dependencies = dependencies;
  }

  get state(): ModelSessionState {
    return this.#state;
  }

  initialize(): Promise<ReadyModelSession> {
    if (this.#initialization !== undefined) {
      return this.#initialization;
    }

    this.#state = "initializing";
    this.#initialization = this.#initializeOnce();
    return this.#initialization;
  }

  async #initializeOnce(): Promise<ReadyModelSession> {
    try {
      const metadata = await this.#dependencies.loadMetadata(this.#metadataUrl);
      const modelUrl = resolveModelUrl(this.#metadataUrl, metadata);
      const session = await this.#dependencies.createSession(modelUrl);

      try {
        validateModelGraph(session, metadata);
      } catch (cause: unknown) {
        try {
          await session.release();
        } catch {
          console.error("ONNX session cleanup failed after graph mismatch");
        }
        throw cause;
      }

      const readySession = Object.freeze({ metadata, session });
      this.#state = "ready";
      return readySession;
    } catch (cause: unknown) {
      this.#state = "failed";
      console.error("Model session initialization failed");
      throw cause;
    }
  }
}

function validateModelGraph(
  session: ModelRuntimeSession,
  metadata: ModelMetadata,
): void {
  validateTensorBoundary(
    "input",
    session.inputNames,
    session.inputMetadata,
    metadata.input,
  );
  validateTensorBoundary(
    "output",
    session.outputNames,
    session.outputMetadata,
    metadata.output,
  );
}

function validateTensorBoundary(
  boundary: ModelGraphBoundary,
  names: readonly string[],
  values: readonly OnnxInferenceSession.ValueMetadata[],
  expected: ExpectedTensorBoundary,
): void {
  const value = values[0];
  if (
    names.length !== 1 ||
    names[0] !== expected.name ||
    values.length !== 1 ||
    value === undefined ||
    value.name !== expected.name ||
    !value.isTensor ||
    value.type !== expected.dataType ||
    !shapesMatch(value.shape, expected.shape)
  ) {
    throw new ModelGraphContractError(boundary);
  }
}

function shapesMatch(
  actual: readonly (number | string)[],
  expected: readonly (number | null)[],
): boolean {
  return (
    actual.length === expected.length &&
    actual.every((dimension, index) => {
      const expectedDimension = expected[index];
      return expectedDimension === null
        ? typeof dimension === "string"
        : dimension === expectedDimension;
    })
  );
}
