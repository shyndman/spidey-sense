import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import {
  calculateBlockedScore,
  score,
  softmaxLogits,
  validateSessionGraph,
  type ScoringSession,
} from "./evaluate-model";
import type { ModelMetadata } from "../src/model/metadata";

const metadata = {
  input: {
    name: "input",
    dataType: "float32",
    shape: [null, 3, 2, 2],
  },
  output: {
    name: "output",
    dataType: "float32",
    shape: [null, 1000],
  },
  classes: {
    blocked: [{ index: 7 }],
  },
} as unknown as ModelMetadata;

const graphSession = (outputShape: readonly (number | string)[] = [
  "batch",
  1000,
]): ScoringSession => ({
  inputNames: ["input"],
  outputNames: ["output"],
  inputMetadata: [
    {
      name: "input",
      isTensor: true,
      type: "float32",
      shape: ["batch", 3, 2, 2],
    },
  ],
  outputMetadata: [
    {
      name: "output",
      isTensor: true,
      type: "float32",
      shape: outputShape,
    },
  ],
  run() {
    return Promise.resolve({});
  },
  async release() {},
});

describe("evaluate-model numeric contracts", () => {
  it("softmaxes extreme logits without non-finite probabilities", () => {
    const logits = new Float32Array(1_000);
    logits[7] = 1_000;
    logits[8] = -1_000;

    const probabilities = softmaxLogits(logits);

    expect(probabilities).toHaveLength(1_000);
    expect(probabilities[7]).toBe(1);
    expect(probabilities.every(Number.isFinite)).toBe(true);
  });

  it("sums only metadata-declared blocked indices", () => {
    const probabilities = new Float32Array(1_000);
    probabilities[7] = 0.25;
    probabilities[8] = 0.75;

    expect(calculateBlockedScore(probabilities, metadata)).toBe(0.25);
  });

  it("accepts dynamic batch graph dimensions and rejects class shape mismatch", () => {
    expect(() => { validateSessionGraph(graphSession(), metadata); }).not.toThrow();
    expect(() =>
      { validateSessionGraph(graphSession(["batch", 999]), metadata); },
    ).toThrow();
  });
  it("emits aggregate lifecycle events at the pending-manifest cadence", async () => {
    const root = await mkdtemp(join(tmpdir(), "evaluation-score-"));
    const paths = {
      root,
      manifests: join(root, "manifests"),
      images: join(root, "images"),
      annotations: join(root, "annotations"),
      scores: join(root, "scores"),
      errors: join(root, "errors"),
      models: join(root, "models"),
    };
    await Promise.all(
      Object.values(paths)
        .slice(1)
        .map((path) => mkdir(path, { recursive: true })),
    );

    const labels = Array.from({ length: 1_000 }, (_, index) => ({
      index,
      synset: `n${String(index).padStart(8, "0")}`,
      label: `label-${String(index)}`,
    }));
    const scoringMetadata = {
      schemaVersion: 2,
      model: {
        id: "evaluation-model",
        filename: "model.onnx",
        sha256: "a".repeat(64),
        sizeBytes: 1,
        format: "onnx",
        opset: 1,
        sourceUrl: "https://example.invalid/model",
        sourceRevision: "test",
      },
      input: {
        name: "input",
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
        name: "output",
        dataType: "float32",
        shape: [null, 1_000],
        activation: "softmax",
        labels,
      },
      classes: {
        blocked: (() => {
          const blockedLabel = labels[7];
          if (blockedLabel === undefined) {
            throw new Error("missing blocked test label");
          }
          return [blockedLabel];
        })(),
        debug: [],
      },
    };
    await writeFile(
      join(paths.models, "model.metadata.json"),
      JSON.stringify(scoringMetadata),
    );

    for (let index = 0; index < 26; index += 1) {
      const sampleId = `sample-${String(index).padStart(3, "0")}`;
      await writeFile(
        join(paths.manifests, `${sampleId}.json`),
        JSON.stringify({
          schema_version: 1,
          sample_id: sampleId,
          source: "inaturalist",
          source_id: "source",
          source_category: "argiope_aurantia",
          expected_presence: "positive",
          source_url: "https://example.invalid/source",
          license: "cc0",
          image_relative_path: `images/${sampleId}.jpg`,
          sha256: "b".repeat(64),
          perceptual_hash: "c".repeat(16),
          duplicate_group: "group",
          split: "calibration",
          width: 2,
          height: 2,
        }),
      );
      await writeFile(join(paths.images, `${sampleId}.jpg`), Buffer.from([0]));
    }

    const session: ScoringSession = {
      inputNames: ["input"],
      outputNames: ["output"],
      inputMetadata: [
        {
          name: "input",
          isTensor: true,
          type: "float32",
          shape: ["batch", 3, 2, 2],
        },
      ],
      outputMetadata: [
        {
          name: "output",
          isTensor: true,
          type: "float32",
          shape: ["batch", 1_000],
        },
      ],
      run() {
        return Promise.resolve({
          output: {
            type: "float32",
            dims: [1, 1_000],
            data: new Float32Array(1_000),
          },
        });
      },
      async release() {},
    };
    const dependencies = {
      decodeJpeg: () =>
        Promise.resolve({
          width: 2,
          height: 2,
          data: new Uint8ClampedArray(new ArrayBuffer(16)),
          channelOrder: "RGBA" as const,
          colorSpace: "srgb" as const,
          alphaMode: "unpremultiplied" as const,
        }),
      transform: () => Promise.resolve(new Float32Array(12)),
      createSession: () => Promise.resolve(session),
    };
    const stderrLines: string[] = [];
    const stderrWrite = vi
      .spyOn(process.stderr, "write")
      .mockImplementation((chunk) => {
        stderrLines.push(chunk.toString());
        return true;
      });

    const summary = await score(paths, dependencies).finally(() => {
      stderrWrite.mockRestore();
    });

    const output = stderrLines.join("");
    const events: Array<Record<string, unknown>> = output
      .split("\n")
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(summary).toEqual({
      schema_version: 1,
      attempted: 26,
      completed: 26,
      skipped: 0,
      failed: 0,
    });
    expect(events).toHaveLength(5);
    expect(events.map((event) => event.event)).toEqual([
      "start",
      "model_loading",
      "model_ready",
      "progress",
      "complete",
    ]);
    expect(events[3]?.processed).toBe(25);
    expect(events[4]?.processed).toBe(26);
    const expectedKeys = [
      "attempted",
      "completed",
      "event",
      "failed",
      "processed",
      "skipped",
      "stage",
    ];
    expect(
      events.every((event) => Object.keys(event).sort().join(",") === expectedKeys.join(",")),
    ).toBe(true);
    expect(output).not.toContain("sample-");
    expect(output).not.toContain(root);
    expect(output).not.toContain("example.invalid");
  });
});
