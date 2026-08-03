import { describe, expect, it } from "vitest";

import {
  loadModelMetadata,
  parseModelMetadata,
  resolveModelUrl,
} from "./metadata";
interface MutableLabel {
  index: number;
  synset: string;
  label: string;
}


function metadataFixture() {
  const labels: [MutableLabel, MutableLabel] = [
    { index: 0, synset: "n00000001", label: "first" },
    { index: 1, synset: "n00000002", label: "second" },
  ];
  const blocked: [MutableLabel] = [{ ...labels[1] }];
  const debug: [MutableLabel] = [{ ...labels[0] }];

  return {
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
      shape: [null, labels.length],
      activation: "softmax",
      labels,
    },
    classes: {
      blocked,
      debug,
    },
  };
}


describe("parseModelMetadata", () => {
  it("parses the generated camelCase metadata shape", () => {
    const parsed = parseModelMetadata(metadataFixture());

    expect(parsed.schemaVersion).toBe(1);
    expect(parsed.output.labels).toHaveLength(2);
    expect(parsed.classes.blocked[0]).toEqual(parsed.output.labels[1]);
  });

  it("rejects unsupported schema versions", () => {
    const value = metadataFixture();
    value.schemaVersion = 2;

    expect(() => parseModelMetadata(value)).toThrow();
  });

  it("rejects unknown keys at every strict object boundary", () => {
    expect(() =>
      parseModelMetadata({ ...metadataFixture(), unexpected: true }),
    ).toThrow();
    expect(() =>
      parseModelMetadata({
        ...metadataFixture(),
        input: { ...metadataFixture().input, unexpected: true },
      }),
    ).toThrow();
  });

  it.each([
    ["negative", -1],
    ["non-integer", 0.5],
    ["noncontiguous", 3],
  ])("rejects %s label indices", (_description, index) => {
    const value = metadataFixture();
    value.output.labels[1].index = index;
    value.classes.blocked[0].index = index;

    expect(() => parseModelMetadata(value)).toThrow();
  });

  it("rejects a label count that differs from the output dimension", () => {
    const value = metadataFixture();

    expect(() =>
      parseModelMetadata({
        ...value,
        output: { ...value.output, shape: [null, 3] },
      }),
    ).toThrow();
  });

  it("rejects duplicate label identities", () => {
    const value = metadataFixture();
    value.output.labels[1].synset = value.output.labels[0].synset;
    value.classes.blocked[0].synset = value.output.labels[0].synset;

    expect(() => parseModelMetadata(value)).toThrow();
  });

  it("rejects class records that do not exactly match the indexed label", () => {
    const value = metadataFixture();
    value.classes.blocked[0].label = "not the output label";

    expect(() => parseModelMetadata(value)).toThrow();
  });

  it("rejects duplicate and overlapping class memberships", () => {
    const duplicate = metadataFixture();
    duplicate.classes.blocked.push({ ...duplicate.classes.blocked[0] });
    const overlap = metadataFixture();
    overlap.classes.debug = [{ ...overlap.classes.blocked[0] }];

    expect(() => parseModelMetadata(duplicate)).toThrow();
    expect(() => parseModelMetadata(overlap)).toThrow();
  });
});

describe("resolveModelUrl", () => {
  it("resolves the model filename relative to the injected metadata URL", () => {
    const metadataUrl = new URL("https://metadata.invalid/releases/current/metadata.json");
    const result = resolveModelUrl(
      metadataUrl,
      parseModelMetadata(metadataFixture()),
    );

    expect(result.href).toBe(
      new URL("model.onnx", metadataUrl).href,
    );
  });
});

describe("loadModelMetadata", () => {
  it("fetches and validates JSON through an injected implementation", async () => {
    const fetcher: typeof fetch = async () =>
      new Response(JSON.stringify(metadataFixture()), {
        status: 200,
        headers: { "content-type": "application/json" },
      });

    await expect(
      loadModelMetadata(new URL("https://metadata.invalid/metadata.json"), fetcher),
    ).resolves.toEqual(parseModelMetadata(metadataFixture()));
  });

  it("rejects non-OK responses", async () => {
    const fetcher: typeof fetch = async () => new Response(null, { status: 503 });

    await expect(
      loadModelMetadata(new URL("https://metadata.invalid/metadata.json"), fetcher),
    ).rejects.toThrow("HTTP 503");
  });

  it("preserves a fetch failure as the public error cause", async () => {
    const networkError = new Error("synthetic network failure");
    const fetcher: typeof fetch = async () => {
      throw networkError;
    };

    await expect(
      loadModelMetadata(new URL("https://metadata.invalid/metadata.json"), fetcher),
    ).rejects.toMatchObject({
      message: "Failed to fetch model metadata",
      cause: networkError,
    });
  });

  it("rejects malformed response JSON through the public API", async () => {
    const fetcher: typeof fetch = async () => new Response("not json");

    await expect(
      loadModelMetadata(new URL("https://metadata.invalid/metadata.json"), fetcher),
    ).rejects.toThrow("Failed to decode model metadata JSON");
  });
});
