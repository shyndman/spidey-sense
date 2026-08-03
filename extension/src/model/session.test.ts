import type { InferenceSession } from "onnxruntime-web/wasm";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseModelMetadata, type ModelMetadata } from "./metadata";
import {
  ModelGraphContractError,
  type ModelRuntimeSession,
  ModelSessionManager,
} from "./session";

function metadataFixture(): ModelMetadata {
  return parseModelMetadata({
    schemaVersion: 1,
    model: {
      id: "synthetic",
      filename: "model.onnx",
      sha256: "0".repeat(64),
      sizeBytes: 16,
      format: "onnx",
      opset: 17,
      sourceUrl: "https://source.invalid/model",
      sourceRevision: "revision",
    },
    input: {
      name: "input",
      dataType: "float32",
      layout: "NCHW",
      shape: [null, 3, 2, 2],
      colorSpace: "RGB",
      resizeMode: "shortest_side",
      resizeShortestSide: 2,
      interpolation: "bilinear",
      cropMode: "center",
      cropWidth: 2,
      cropHeight: 2,
      pixelScale: 1 / 255,
      mean: [0.1, 0.2, 0.3],
      standardDeviation: [0.4, 0.5, 0.6],
    },
    output: {
      name: "output",
      dataType: "float32",
      shape: [null, 2],
      activation: "softmax",
      labels: [
        { index: 0, synset: "n00000001", label: "first" },
        { index: 1, synset: "n00000002", label: "second" },
      ],
    },
    classes: {
      blocked: [{ index: 1, synset: "n00000002", label: "second" }],
      debug: [{ index: 0, synset: "n00000001", label: "first" }],
    },
  });
}

function runtimeSession(
  overrides: Partial<ModelRuntimeSession> = {},
): ModelRuntimeSession {
  const run = vi.fn(async (): Promise<InferenceSession.ReturnType> => ({}));
  const release = vi.fn(async (): Promise<void> => undefined);

  return {
    inputNames: ["input"],
    outputNames: ["output"],
    inputMetadata: [
      {
        name: "input",
        isTensor: true,
        type: "float32",
        shape: ["batch_size", 3, 2, 2],
      },
    ],
    outputMetadata: [
      {
        name: "output",
        isTensor: true,
        type: "float32",
        shape: ["batch_size", 2],
      },
    ],
    run,
    release,
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ModelSessionManager", () => {
  it("loads one validated model session for concurrent and repeated callers", async () => {
    const metadata = metadataFixture();
    const session = runtimeSession();
    const loadMetadata = vi.fn(async () => metadata);
    const createSession = vi.fn(async () => session);
    const manager = new ModelSessionManager(
      new URL("moz-extension://extension/models/model.metadata.json"),
      { loadMetadata, createSession },
    );

    const first = manager.initialize();
    const second = manager.initialize();

    expect(manager.state).toBe("initializing");
    expect(second).toBe(first);
    await expect(first).resolves.toEqual({ metadata, session });
    expect(manager.state).toBe("ready");
    expect(loadMetadata).toHaveBeenCalledExactlyOnceWith(
      new URL("moz-extension://extension/models/model.metadata.json"),
    );
    expect(createSession).toHaveBeenCalledExactlyOnceWith(
      new URL("moz-extension://extension/models/model.onnx"),
    );
    expect(manager.initialize()).toBe(first);
    expect(createSession).toHaveBeenCalledTimes(1);
  });

  it.each([
    {
      description: "input name",
      boundary: "input",
      overrides: { inputNames: ["wrong-input"] },
    },
    {
      description: "input tensor type",
      boundary: "input",
      overrides: {
        inputMetadata: [
          {
            name: "input",
            isTensor: true,
            type: "int32",
            shape: ["batch_size", 3, 2, 2],
          },
        ],
      },
    },
    {
      description: "output dimensions",
      boundary: "output",
      overrides: {
        outputMetadata: [
          {
            name: "output",
            isTensor: true,
            type: "float32",
            shape: ["batch_size", 3],
          },
        ],
      },
    },
  ] as const)(
    "rejects a mismatched $description before inference",
    async ({ boundary, overrides }) => {
      const session = runtimeSession(overrides);
      const manager = new ModelSessionManager(
        new URL("moz-extension://extension/models/model.metadata.json"),
        {
          loadMetadata: async () => metadataFixture(),
          createSession: async () => session,
        },
      );
      vi.spyOn(console, "error").mockImplementation(() => undefined);

      const initialization = manager.initialize();

      await expect(initialization).rejects.toMatchObject({
        name: "ModelGraphContractError",
        code: "GRAPH_MISMATCH",
        boundary,
      } satisfies Partial<ModelGraphContractError>);
      expect(session.release).toHaveBeenCalledExactlyOnceWith();
      expect(session.run).not.toHaveBeenCalled();
      expect(manager.state).toBe("failed");
    },
  );

  it("retains one failed attempt and traces no failure details", async () => {
    const cause = new Error("private model location and runtime details");
    const createSession = vi.fn(async (): Promise<ModelRuntimeSession> => {
      throw cause;
    });
    const manager = new ModelSessionManager(
      new URL("moz-extension://private/models/model.metadata.json"),
      {
        loadMetadata: async () => metadataFixture(),
        createSession,
      },
    );
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);

    const first = manager.initialize();
    await expect(first).rejects.toBe(cause);

    expect(manager.state).toBe("failed");
    expect(manager.initialize()).toBe(first);
    await expect(manager.initialize()).rejects.toBe(cause);
    expect(createSession).toHaveBeenCalledTimes(1);
    expect(errorLog).toHaveBeenCalledExactlyOnceWith(
      "Model session initialization failed",
    );
  });
});
