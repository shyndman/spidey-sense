import { afterEach, describe, expect, it, vi } from "vitest";

import type { ModelMetadata } from "../model/metadata";
import { parseModelMetadata } from "../model/metadata";
import type { DecodedImage } from "./decoding";
import {
  type ImagePreprocessingDependencies,
  ImagePreprocessingErrorCode,
  transformImageToModelInput,
} from "./preprocessing";

const RGBA_CHANNEL_COUNT = 4;
const OPAQUE_ALPHA = 255;

function metadataFixture(
  inputOverrides: Partial<ModelMetadata["input"]> = {},
): ModelMetadata {
  const input = {
    name: "input",
    dataType: "float32",
    layout: "NCHW",
    shape: [null, 3, 2, 2],
    colorSpace: "RGB",
    resizeMode: "contain",
    allowUpscale: true,
    interpolation: "bilinear",
    paddingMode: "black",
    pixelScale: 1 / 255,
    mean: [0.1, 0.2, 0.3],
    standardDeviation: [0.5, 0.25, 1],
    ...inputOverrides,
  };
  return parseModelMetadata({
    schemaVersion: 2,
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
    input,
    output: {
      name: "output",
      dataType: "float32",
      shape: [null, 1],
      activation: "softmax",
      labels: [{ index: 0, synset: "n00000001", label: "proxy" }],
    },
    classes: { blocked: [], debug: [] },
  });
}

function decodedImage(
  width: number,
  height: number,
  data: Uint8ClampedArray<ArrayBuffer>,
): DecodedImage {
  return {
    width,
    height,
    data,
    channelOrder: "RGBA",
    colorSpace: "srgb",
    alphaMode: "unpremultiplied",
  };
}

function emptyRgba(width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  return new Uint8ClampedArray(width * height * RGBA_CHANNEL_COUNT);
}

function writePixel(
  data: Uint8ClampedArray<ArrayBuffer>,
  width: number,
  x: number,
  y: number,
  rgba: readonly [number, number, number, number],
): void {
  const offset = (y * width + x) * RGBA_CHANNEL_COUNT;
  data.set(rgba, offset);
}

function expectFloatArrayClose(
  actual: Float32Array,
  expected: readonly number[],
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const expectedValue = expected.at(index);
    if (expectedValue === undefined) {
      throw new Error(`Missing expected value at index ${String(index)}`);
    }
    expect(actual[index]).toBeCloseTo(expectedValue, 6);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("transformImageToModelInput", () => {
  it("contains the full image, pads black, normalizes, and writes NCHW", async () => {
    const debugLog = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const durationMatcher: unknown = expect.any(Number);
    const image = decodedImage(2, 1, emptyRgba(2, 1));
    const resizedData = emptyRgba(2, 1);
    writePixel(resizedData, 2, 0, 0, [255, 0, 0, OPAQUE_ALPHA]);
    writePixel(resizedData, 2, 1, 0, [0, 255, 0, 128]);
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>(() =>
      Promise.resolve({ width: 2, height: 1, data: resizedData }),
    );
    const metadata = metadataFixture();

    const input = await transformImageToModelInput(image, metadata, { resize });

    expect(resize).toHaveBeenCalledExactlyOnceWith(image, 2, 1);
    const halfGreen = 128 / 255;
    expectFloatArrayClose(input, [
      1.8,
      -0.2,
      -0.2,
      -0.2,
      -0.8,
      (halfGreen - 0.2) / 0.25,
      -0.8,
      -0.8,
      -0.3,
      -0.3,
      -0.3,
      -0.3,
    ]);
    expect(debugLog).toHaveBeenCalledExactlyOnceWith(
      "Decoded image transformed into the model input boundary",
      {
        durationMilliseconds: durationMatcher,
        sourceWidth: 2,
        sourceHeight: 1,
        resizedWidth: 2,
        resizedHeight: 1,
        paddingLeft: 0,
        paddingTop: 0,
        outputElements: 12,
      },
    );
  });

  it("preserves aspect ratio within the complete model boundary", async () => {
    vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const image = decodedImage(4, 3, emptyRgba(4, 3));
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>(() =>
      Promise.resolve({ width: 3, height: 2, data: emptyRgba(3, 2) }),
    );
    const metadata = metadataFixture({ shape: [null, 3, 2, 3] });

    await transformImageToModelInput(image, metadata, { resize });

    expect(resize).toHaveBeenCalledExactlyOnceWith(image, 3, 2);
  });

  it("centers contained pixels with odd padding on the trailing edge", async () => {
    vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const image = decodedImage(1, 2, emptyRgba(1, 2));
    const resizedData = emptyRgba(2, 3);
    for (let y = 0; y < 3; y += 1) {
      writePixel(resizedData, 2, 0, y, [1, 0, 0, OPAQUE_ALPHA]);
      writePixel(resizedData, 2, 1, y, [2, 0, 0, OPAQUE_ALPHA]);
    }
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>(() =>
      Promise.resolve({ width: 2, height: 3, data: resizedData }),
    );
    const metadata = metadataFixture({
      shape: [null, 3, 3, 3],
      pixelScale: 1,
      mean: [0, 0, 0],
      standardDeviation: [1, 1, 1],
    });

    const input = await transformImageToModelInput(image, metadata, { resize });

    expect(Array.from(input.slice(0, 9))).toEqual([
      1, 2, 0,
      1, 2, 0,
      1, 2, 0,
    ]);
  });

  it("rejects an invalid decoded pixel buffer before resizing", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>();
    const image = decodedImage(2, 2, new Uint8ClampedArray([1, 2, 3]));

    await expect(
      transformImageToModelInput(image, metadataFixture(), { resize }),
    ).rejects.toMatchObject({
      code: ImagePreprocessingErrorCode.INVALID_IMAGE,
      message: "Decoded image does not match the RGBA8 pixel contract",
    });
    expect(resize).not.toHaveBeenCalled();
  });

  it("rejects an invalid resizer result", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>(() =>
      Promise.resolve({ width: 2, height: 1, data: new Uint8ClampedArray() }),
    );

    await expect(
      transformImageToModelInput(
        decodedImage(2, 1, emptyRgba(2, 1)),
        metadataFixture(),
        { resize },
      ),
    ).rejects.toMatchObject({
      code: ImagePreprocessingErrorCode.INVALID_RESIZE_OUTPUT,
      message: "Image resizer returned an invalid RGBA8 result",
    });
  });

  it("rejects float32 overflow at the tensor boundary", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const resizedData = emptyRgba(2, 1);
    resizedData.fill(OPAQUE_ALPHA);
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>(() =>
      Promise.resolve({ width: 2, height: 1, data: resizedData }),
    );
    const metadata = metadataFixture({
      pixelScale: Number.MAX_VALUE,
      mean: [0, 0, 0],
      standardDeviation: [1, 1, 1],
    });

    await expect(
      transformImageToModelInput(
        decodedImage(2, 1, emptyRgba(2, 1)),
        metadata,
        { resize },
      ),
    ).rejects.toMatchObject({
      code: ImagePreprocessingErrorCode.NON_FINITE_OUTPUT,
      message: "Image transform produced a non-finite tensor value",
    });
  });

  it("wraps resizer failures without logging their contents", async () => {
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const durationMatcher: unknown = expect.any(Number);
    const resize = vi.fn<ImagePreprocessingDependencies["resize"]>(() =>
      Promise.reject(new Error("sensitive-pixel-sentinel")),
    );

    await expect(
      transformImageToModelInput(
        decodedImage(2, 1, emptyRgba(2, 1)),
        metadataFixture(),
        { resize },
      ),
    ).rejects.toMatchObject({
      code: ImagePreprocessingErrorCode.TRANSFORMATION_FAILED,
      message: "Image transform failed unexpectedly",
    });
    expect(errorLog).toHaveBeenCalledExactlyOnceWith(
      "Image preprocessing stopped: TRANSFORMATION_FAILED",
      {
        durationMilliseconds: durationMatcher,
        sourceWidth: 2,
        sourceHeight: 1,
      },
    );
  });
});
