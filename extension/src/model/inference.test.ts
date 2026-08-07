import { Tensor, type InferenceSession } from 'onnxruntime-web/wasm';
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  InferenceError,
  InferenceErrorCode,
  type InferenceSessionLike,
  runModelInference,
} from "./inference";
import type { ModelMetadata } from "./metadata";

const metadata: ModelMetadata = {
  schemaVersion: 2,
  model: {
    id: "synthetic-model",
    filename: "synthetic.onnx",
    sha256: "a".repeat(64),
    sizeBytes: 1,
    format: "onnx",
    opset: 18,
    sourceUrl: "https://example.invalid/model.onnx",
    sourceRevision: "test-revision",
  },
  input: {
    name: "metadata_input",
    dataType: "float32",
    layout: "NCHW",
    shape: [null, 3, 2, 2],
    colorSpace: "RGB",
    resizeMode: "contain",
    allowUpscale: true,
    interpolation: "bilinear",
    paddingMode: "black",
    pixelScale: 1,
    mean: [0, 0, 0],
    standardDeviation: [1, 1, 1],
  },
  output: {
    name: "metadata_output",
    dataType: "float32",
    shape: [null, 3],
    activation: "softmax",
    labels: [
      { index: 0, synset: "n00000001", label: "first" },
      { index: 1, synset: "n00000002", label: "second" },
      { index: 2, synset: "n00000003", label: "third" },
    ],
  },
  classes: {
    blocked: [{ index: 1, synset: "n00000002", label: "second" }],
    debug: [
      { index: 0, synset: "n00000001", label: "first" },
      { index: 2, synset: "n00000003", label: "third" },
    ],
  },
};

function fakeSession(
  run: InferenceSessionLike["run"],
): InferenceSessionLike {
  return { run };
}

function outputTensor(
  data: Float32Array,
  dimensions: readonly number[] = [1, 3],
): Tensor {
  return new Tensor("float32", data, dimensions);
}

async function expectInferenceCode(
  promise: Promise<unknown>,
  code: InferenceErrorCode,
): Promise<InferenceError> {
  try {
    await promise;
  } catch (error: unknown) {
    expect(error).toBeInstanceOf(InferenceError);
    expect((error as InferenceError).code).toBe(code);
    return error as InferenceError;
  }
  throw new Error("Expected inference to fail");
}

describe("runModelInference", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "debug").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("normalizes stable probabilities and returns one prediction plus class sums", async () => {
    const session = fakeSession(() =>
      Promise.resolve({
        metadata_output: outputTensor(new Float32Array([1_000, 1_002, 1_001])),
      }),
    );
    const debugLog = vi.spyOn(console, "debug");

    const result = await runModelInference(
      session,
      metadata,
      new Float32Array(12),
    );

    expect([...result.probabilities]).toEqual([
      expect.closeTo(0.09003057, 6),
      expect.closeTo(0.66524096, 6),
      expect.closeTo(0.24472847, 6),
    ]);
    expect(
      result.probabilities.reduce((sum, probability) => sum + probability, 0),
    ).toBeCloseTo(1);
    const { probability, ...topPrediction } = result.topPrediction;
    expect(topPrediction).toEqual({
      index: 1,
      synset: "n00000002",
      label: "second",
    });
    expect(probability).toBeCloseTo(0.66524096, 6);
    expect(result.blockedScore).toBeCloseTo(0.66524096);
    expect(result.debugScore).toBeCloseTo(0.33475904);
    expect(debugLog).toHaveBeenCalledExactlyOnceWith(
      "Model inference completed",
      expect.objectContaining({
        inputElements: 12,
        outputElements: 3,
      }),
    );
    expect(debugLog.mock.calls[0]?.[1]).toHaveProperty(
      "durationMilliseconds",
      expect.any(Number),
    );
  });

  it("uses metadata graph names for the feed and selected output", async () => {
    const run = vi.fn<InferenceSessionLike["run"]>((feeds, fetches) => {
      expect(Object.keys(feeds)).toEqual(["metadata_input"]);
      expect(feeds.metadata_input).toBeInstanceOf(Tensor);
      expect((feeds.metadata_input as Tensor).dims).toEqual([1, 3, 2, 2]);
      expect(fetches).toEqual(["metadata_output"]);
      return Promise.resolve({
        metadata_output: outputTensor(new Float32Array([0, 1, 2])),
      });
    });
    await runModelInference(fakeSession(run), metadata, new Float32Array(12));
    expect(run).toHaveBeenCalledOnce();
  });

  it("rejects the wrong input element count", async () => {
    const run = vi.fn<InferenceSessionLike["run"]>();
    await expectInferenceCode(
      runModelInference(fakeSession(run), metadata, new Float32Array(11)),
      InferenceErrorCode.INVALID_INPUT,
    );
    expect(run).not.toHaveBeenCalled();
  });

  it.each([
    ["missing", {}],
    ["wrong value kind", { metadata_output: { type: "float32" } }],
  ])("rejects a %s metadata-defined output", async (_caseName, outputs) => {
    const session = fakeSession(() =>
      Promise.resolve(outputs as InferenceSession.ReturnType),
    );
    await expectInferenceCode(
      runModelInference(session, metadata, new Float32Array(12)),
      InferenceErrorCode.INVALID_OUTPUT,
    );
  });

  it.each([
    ["wrong rank", outputTensor(new Float32Array([0, 1, 2]), [3])],
    ["wrong batch", outputTensor(new Float32Array(6), [2, 3])],
    ["wrong class dimension", outputTensor(new Float32Array(2), [1, 2])],
    [
      "wrong element type",
      new Tensor("int32", new Int32Array([0, 1, 2]), [1, 3]),
    ],
  ])("rejects output with %s", async (_caseName, output) => {
    const session = fakeSession(() =>
      Promise.resolve({ metadata_output: output }),
    );
    await expectInferenceCode(
      runModelInference(session, metadata, new Float32Array(12)),
      InferenceErrorCode.INVALID_OUTPUT,
    );
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects the non-finite logit %s",
    async (nonFiniteLogit) => {
      const session = fakeSession(() =>
        Promise.resolve({
          metadata_output: outputTensor(
            new Float32Array([0, nonFiniteLogit, 2]),
          ),
        }),
      );
      await expectInferenceCode(
        runModelInference(session, metadata, new Float32Array(12)),
        InferenceErrorCode.NON_FINITE_OUTPUT,
      );
    },
  );

  it("wraps and preserves a session rejection", async () => {
    const runtimeCause = new Error("synthetic runtime rejection");
    const session = fakeSession(() => Promise.reject(runtimeCause));

    const error = await expectInferenceCode(
      runModelInference(session, metadata, new Float32Array(12)),
      InferenceErrorCode.RUNTIME_EXECUTION_FAILED,
    );
    expect(error.cause).toBe(runtimeCause);
    const errorLog = vi.spyOn(console, "error");
    expect(errorLog).toHaveBeenCalledWith(
      "Model inference failed",
      expect.objectContaining({
        code: InferenceErrorCode.RUNTIME_EXECUTION_FAILED,
        inputElements: 12,
      }),
    );
    expect(errorLog.mock.calls[0]?.[1]).toHaveProperty(
      "durationMilliseconds",
      expect.any(Number),
    );
  });
});
