import { Tensor, type InferenceSession } from 'onnxruntime-web/wasm';

import type { ModelLabel, ModelMetadata } from "./metadata";

export const InferenceErrorCode = {
  INVALID_INPUT: "INVALID_INPUT",
  RUNTIME_EXECUTION_FAILED: "RUNTIME_EXECUTION_FAILED",
  INVALID_OUTPUT: "INVALID_OUTPUT",
  NON_FINITE_OUTPUT: "NON_FINITE_OUTPUT",
} as const;

export type InferenceErrorCode =
  (typeof InferenceErrorCode)[keyof typeof InferenceErrorCode];

export class InferenceError extends Error {
  readonly code: InferenceErrorCode;

  constructor(code: InferenceErrorCode, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "InferenceError";
    this.code = code;
  }
}

/**
 * The only part of an ONNX Runtime session needed at the user-facing inference
 * boundary. Keeping this structural makes a real `onnxruntime-web` session
 * assignable while allowing tests and embedders to supply a deterministic
 * implementation without constructing a runtime backend.
 */
export interface InferenceSessionLike {
  run(
    feeds: InferenceSession.FeedsType,
    fetches: readonly string[],
  ): Promise<InferenceSession.ReturnType>;
}

export interface TopPrediction extends ModelLabel {
  readonly probability: number;
}

export interface InferenceResult {
  /** Probabilities in exactly the same index order as `metadata.output.labels`. */
  readonly probabilities: Float32Array;
  /** The single highest-probability label; ties retain the lowest label index. */
  readonly topPrediction: TopPrediction;
  /** Sum of probabilities for metadata's blocked class group. Not a decision. */
  readonly blockedScore: number;
  /** Sum of probabilities for metadata's debug class group. */
  readonly debugScore: number;
}

/**
 * Runs one already-preprocessed NCHW sample through a ready ONNX Runtime
 * session. This is intentionally a numeric boundary only: callers own model
 * loading and image preprocessing, while later policy code owns any blocking
 * decision. Graph names, dimensions, labels, activation, and semantic groups
 * all come from previously validated metadata so this module duplicates no
 * model constants.
 *
 * Logs expose only elapsed time, input/output element counts, and stable
 * failure codes. Tensor values, model output, model bytes, image data, and URLs
 * are never included. The thrown `InferenceError` retains the runtime cause for
 * programmatic diagnostics.
 */
export async function runModelInference(
  session: InferenceSessionLike,
  metadata: ModelMetadata,
  input: Float32Array,
): Promise<InferenceResult> {
  const startedAt = performance.now();
  try {
    const result = await executeInference(session, metadata, input);
    console.debug("Model inference completed", {
      durationMilliseconds: performance.now() - startedAt,
      inputElements: input.length,
      outputElements: result.probabilities.length,
    });
    return result;
  } catch (cause: unknown) {
    const error =
      cause instanceof InferenceError
        ? cause
        : new InferenceError(
            InferenceErrorCode.INVALID_OUTPUT,
            "Unexpected model inference failure",
            cause,
          );

    console.error("Model inference failed", {
      code: error.code,
      durationMilliseconds: performance.now() - startedAt,
      inputElements: input.length,
    });
    throw error;
  }
}

async function executeInference(
  session: InferenceSessionLike,
  metadata: ModelMetadata,
  input: Float32Array,
): Promise<InferenceResult> {
  const inputTensor = createInputTensor(metadata, input);
  const outputs = await runSession(session, metadata, inputTensor);
  const logits = validateOutput(
    outputs[metadata.output.name],
    metadata.output.labels.length,
  );
  const probabilities = softmax(logits);
  const topIndex = findTopIndex(probabilities);
  const topLabel = requireValue(
    metadata.output.labels[topIndex],
    "The output tensor does not contain a top prediction label",
  );
  const topProbability = requireValue(
    probabilities[topIndex],
    "The output tensor does not contain a top prediction probability",
  );

  return {
    probabilities,
    topPrediction: { ...topLabel, probability: topProbability },
    blockedScore: sumClassGroup(probabilities, metadata.classes.blocked),
    debugScore: sumClassGroup(probabilities, metadata.classes.debug),
  };
}

function createInputTensor(
  metadata: ModelMetadata,
  input: Float32Array,
): Tensor {
  const inputDimensions = [
    1,
    metadata.input.shape[1],
    metadata.input.shape[2],
    metadata.input.shape[3],
  ] as const;
  const expectedInputLength =
    inputDimensions[1] * inputDimensions[2] * inputDimensions[3];

  if (input.length !== expectedInputLength) {
    throw new InferenceError(
      InferenceErrorCode.INVALID_INPUT,
      `Expected ${String(expectedInputLength)} float32 input values, received ${String(input.length)}`,
    );
  }

  return new Tensor("float32", input, inputDimensions);
}

async function runSession(
  session: InferenceSessionLike,
  metadata: ModelMetadata,
  input: Tensor,
): Promise<InferenceSession.ReturnType> {
  try {
    return await session.run(
      { [metadata.input.name]: input },
      [metadata.output.name],
    );
  } catch (cause: unknown) {
    throw new InferenceError(
      InferenceErrorCode.RUNTIME_EXECUTION_FAILED,
      "ONNX Runtime failed to execute the model",
      cause,
    );
  }
}

function validateOutput(
  output: Tensor | undefined,
  labelCount: number,
): Float32Array {
  if (
    !(output instanceof Tensor) ||
    output.type !== "float32" ||
    output.dims.length !== 2 ||
    output.dims[0] !== 1 ||
    output.dims[1] !== labelCount ||
    !(output.data instanceof Float32Array) ||
    output.data.length !== labelCount
  ) {
    throw new InferenceError(
      InferenceErrorCode.INVALID_OUTPUT,
      "The output tensor does not match the metadata-defined float32 batch-1 boundary",
    );
  }

  return output.data;
}

function softmax(logits: Float32Array): Float32Array {
  let maximumLogit = Number.NEGATIVE_INFINITY;
  for (const logit of logits) {
    if (!Number.isFinite(logit)) {
      throw new InferenceError(
        InferenceErrorCode.NON_FINITE_OUTPUT,
        "The output tensor contains a non-finite logit",
      );
    }
    if (logit > maximumLogit) maximumLogit = logit;
  }
  const probabilities = new Float32Array(logits.length);
  let exponentialSum = 0;

  for (let index = 0; index < logits.length; index += 1) {
    const logit = requireValue(
      logits[index],
      "The output tensor is missing a required logit",
    );
    const exponential = Math.exp(logit - maximumLogit);
    probabilities[index] = exponential;
    exponentialSum += exponential;
  }
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = requireValue(
      probabilities[index],
      "The output tensor is missing a required probability",
    );
    probabilities[index] = probability / exponentialSum;
  }
  return probabilities;
}

function findTopIndex(probabilities: Float32Array): number {
  let topIndex = 0;
  let topProbability = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = requireValue(
      probabilities[index],
      "The output tensor is missing a required probability",
    );
    if (probability > topProbability) {
      topProbability = probability;
      topIndex = index;
    }
  }
  return topIndex;
}

function sumClassGroup(
  probabilities: Float32Array,
  labels: readonly ModelLabel[],
): number {
  let sum = 0;
  for (const label of labels) {
    sum += requireValue(
      probabilities[label.index],
      "The output tensor is missing a required class probability",
    );
  }
  return sum;
}

function requireValue<T>(value: T | undefined, message: string): T {
  if (value === undefined) {
    throw new InferenceError(InferenceErrorCode.INVALID_OUTPUT, message);
  }
  return value;
}
