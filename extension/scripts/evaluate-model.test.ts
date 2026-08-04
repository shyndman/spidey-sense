import { describe, expect, it } from "vitest";

import {
  calculateBlockedScore,
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
  async run() {
    return {};
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
    expect(() => validateSessionGraph(graphSession(), metadata)).not.toThrow();
    expect(() =>
      validateSessionGraph(graphSession(["batch", 999]), metadata),
    ).toThrow();
  });
});
