//! HACK: @jsquash/resize's public entrypoint statically imports three WASM
//! engines even when only its triangle resizer is used. Importing the pinned
//! package's generated resize binding directly keeps the production XPI from
//! carrying unused HQX and magic-kernel runtimes. The numeric filter index below
//! is therefore coupled to @jsquash/resize 2.1.1 and must be reviewed with any
//! dependency upgrade.
import initResizeWasm, {
  resize as resizeRgba,
} from "@jsquash/resize/lib/resize/pkg/squoosh_resize.js";

import type { ModelMetadata } from "../model/metadata";
import type { DecodedImage } from "./decoding";

const RGBA_CHANNEL_COUNT = 4;
const RGB_CHANNEL_INDICES = [0, 1, 2] as const;
const ALPHA_CHANNEL_INDEX = 3;
const MAX_CHANNEL_VALUE = 255;
const TRIANGLE_FILTER_INDEX = 0;
const PREMULTIPLY_ALPHA = true;
const CONVERT_TO_LINEAR_RGB = false;

let resizeWasmInitialization: Promise<unknown> | undefined;

export const ImagePreprocessingErrorCode = {
  INVALID_IMAGE: "INVALID_IMAGE",
  INVALID_METADATA: "INVALID_METADATA",
  INVALID_RESIZE_OUTPUT: "INVALID_RESIZE_OUTPUT",
  NON_FINITE_OUTPUT: "NON_FINITE_OUTPUT",
  TRANSFORMATION_FAILED: "TRANSFORMATION_FAILED",
} as const;

export type ImagePreprocessingErrorCode =
  (typeof ImagePreprocessingErrorCode)[keyof typeof ImagePreprocessingErrorCode];

const IMAGE_PREPROCESSING_ERROR_MESSAGES: Readonly<
  Record<ImagePreprocessingErrorCode, string>
> = {
  INVALID_IMAGE: "Decoded image does not match the RGBA8 pixel contract",
  INVALID_METADATA: "Model metadata does not define a valid image transform",
  INVALID_RESIZE_OUTPUT: "Image resizer returned an invalid RGBA8 result",
  NON_FINITE_OUTPUT: "Image transform produced a non-finite tensor value",
  TRANSFORMATION_FAILED: "Image transform failed unexpectedly",
};

/** A stable, content-free failure while producing the model input tensor. */
export class ImagePreprocessingError extends Error {
  readonly code: ImagePreprocessingErrorCode;

  constructor(code: ImagePreprocessingErrorCode, cause?: unknown) {
    super(
      IMAGE_PREPROCESSING_ERROR_MESSAGES[code],
      cause === undefined ? undefined : { cause },
    );
    this.name = "ImagePreprocessingError";
    this.code = code;
  }
}

export interface ResizedRgbaImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray<ArrayBufferLike>;
}

export interface ImagePreprocessingDependencies {
  readonly resize: (
    image: DecodedImage,
    width: number,
    height: number,
  ) => Promise<ResizedRgbaImage>;
}

interface ResizeGeometry {
  readonly width: number;
  readonly height: number;
  readonly cropLeft: number;
  readonly cropTop: number;
}

async function initializeResizeWasm(): Promise<void> {
  resizeWasmInitialization ??= initResizeWasm();
  await resizeWasmInitialization;
}

const defaultDependencies: ImagePreprocessingDependencies = {
  resize: async (image, width, height) => {
    await initializeResizeWasm();
    const input = new Uint8Array(
      image.data.buffer,
      image.data.byteOffset,
      image.data.byteLength,
    );
    const data = resizeRgba(
      input,
      image.width,
      image.height,
      width,
      height,
      TRIANGLE_FILTER_INDEX,
      PREMULTIPLY_ALPHA,
      CONVERT_TO_LINEAR_RGB,
    );
    return { width, height, data };
  },
};

/**
 * Converts decoded RGBA8 sRGB pixels into the exact single-sample NCHW float32
 * input described by generated model metadata. The shorter side is resized with
 * an antialiased bilinear triangle filter, then cropped using Torchvision's
 * integer center geometry. Premultiplied-alpha resizing prevents hidden colors
 * in transparent pixels from bleeding into neighbors; the resized pixels are
 * composited against black before RGB scaling and channel normalization.
 *
 * Transformation is fail-closed at this boundary: every invalid image,
 * malformed transform, resize failure, and non-finite output throws a typed
 * error. Logs contain only stable codes—never pixels, tensor values, dimensions,
 * metadata values, or URLs.
 */
export async function transformImageToModelInput(
  image: DecodedImage,
  metadata: ModelMetadata,
  dependencies: ImagePreprocessingDependencies = defaultDependencies,
): Promise<Float32Array> {
  try {
    validateImage(image);
    validateNormalization(metadata);
    const geometry = deriveResizeGeometry(image, metadata);
    const resized = await dependencies.resize(
      image,
      geometry.width,
      geometry.height,
    );
    validateResizeOutput(resized, geometry);
    const output = normalizeCenterCrop(resized, metadata, geometry);
    console.debug("Decoded image transformed into the model input boundary");
    return output;
  } catch (cause: unknown) {
    const error =
      cause instanceof ImagePreprocessingError
        ? cause
        : new ImagePreprocessingError(
            ImagePreprocessingErrorCode.TRANSFORMATION_FAILED,
            cause,
          );
    console.error(`Image preprocessing stopped: ${error.code}`);
    throw error;
  }
}

function validateImage(image: DecodedImage): void {
  if (
    !Number.isSafeInteger(image.width) ||
    image.width <= 0 ||
    !Number.isSafeInteger(image.height) ||
    image.height <= 0 ||
    image.channelOrder !== "RGBA" ||
    image.colorSpace !== "srgb" ||
    image.alphaMode !== "unpremultiplied"
  ) {
    throw new ImagePreprocessingError(
      ImagePreprocessingErrorCode.INVALID_IMAGE,
    );
  }

  const pixelCount = image.width * image.height;
  if (
    !Number.isSafeInteger(pixelCount) ||
    image.data.length !== pixelCount * RGBA_CHANNEL_COUNT
  ) {
    throw new ImagePreprocessingError(
      ImagePreprocessingErrorCode.INVALID_IMAGE,
    );
  }
}

function validateNormalization(metadata: ModelMetadata): void {
  if (
    !Number.isFinite(metadata.input.pixelScale) ||
    metadata.input.pixelScale <= 0
  ) {
    throw new ImagePreprocessingError(
      ImagePreprocessingErrorCode.INVALID_METADATA,
    );
  }
  for (const channel of RGB_CHANNEL_INDICES) {
    if (
      !Number.isFinite(metadata.input.mean[channel]) ||
      !Number.isFinite(metadata.input.standardDeviation[channel]) ||
      metadata.input.standardDeviation[channel] <= 0
    ) {
      throw new ImagePreprocessingError(
        ImagePreprocessingErrorCode.INVALID_METADATA,
      );
    }
  }
}

function deriveResizeGeometry(
  image: DecodedImage,
  metadata: ModelMetadata,
): ResizeGeometry {
  const targetShortSide = metadata.input.resizeShortestSide;
  const sourceShortSide = Math.min(image.width, image.height);
  const sourceLongSide = Math.max(image.width, image.height);
  const resizedLongSide = Math.trunc(
    (targetShortSide * sourceLongSide) / sourceShortSide,
  );
  const widthIsShorter = image.width <= image.height;
  const width = widthIsShorter ? targetShortSide : resizedLongSide;
  const height = widthIsShorter ? resizedLongSide : targetShortSide;

  if (
    width < metadata.input.cropWidth ||
    height < metadata.input.cropHeight
  ) {
    throw new ImagePreprocessingError(
      ImagePreprocessingErrorCode.INVALID_METADATA,
    );
  }

  return {
    width,
    height,
    cropLeft: roundHalfToEven((width - metadata.input.cropWidth) / 2),
    cropTop: roundHalfToEven((height - metadata.input.cropHeight) / 2),
  };
}

function validateResizeOutput(
  resized: ResizedRgbaImage,
  geometry: ResizeGeometry,
): void {
  const pixelCount = resized.width * resized.height;
  if (
    resized.width !== geometry.width ||
    resized.height !== geometry.height ||
    !Number.isSafeInteger(pixelCount) ||
    resized.data.length !== pixelCount * RGBA_CHANNEL_COUNT
  ) {
    throw new ImagePreprocessingError(
      ImagePreprocessingErrorCode.INVALID_RESIZE_OUTPUT,
    );
  }
}

function normalizeCenterCrop(
  resized: ResizedRgbaImage,
  metadata: ModelMetadata,
  geometry: ResizeGeometry,
): Float32Array {
  const cropWidth = metadata.input.cropWidth;
  const cropHeight = metadata.input.cropHeight;
  const channelPlaneLength = cropWidth * cropHeight;
  const output = new Float32Array(
    RGB_CHANNEL_INDICES.length * channelPlaneLength,
  );

  for (let y = 0; y < cropHeight; y += 1) {
    const sourceY = geometry.cropTop + y;
    for (let x = 0; x < cropWidth; x += 1) {
      const sourceX = geometry.cropLeft + x;
      const sourceOffset =
        (sourceY * resized.width + sourceX) * RGBA_CHANNEL_COUNT;
      const outputOffset = y * cropWidth + x;
      writeNormalizedPixel(
        output,
        outputOffset,
        channelPlaneLength,
        resized.data,
        sourceOffset,
        metadata,
      );
    }
  }
  return output;
}

function writeNormalizedPixel(
  output: Float32Array,
  outputOffset: number,
  channelPlaneLength: number,
  pixels: Uint8ClampedArray<ArrayBufferLike>,
  sourceOffset: number,
  metadata: ModelMetadata,
): void {
  const alpha =
    pixels[sourceOffset + ALPHA_CHANNEL_INDEX]! / MAX_CHANNEL_VALUE;
  for (const channel of RGB_CHANNEL_INDICES) {
    const composited = pixels[sourceOffset + channel]! * alpha;
    const normalized =
      (composited * metadata.input.pixelScale - metadata.input.mean[channel]) /
      metadata.input.standardDeviation[channel];
    const outputIndex = channel * channelPlaneLength + outputOffset;
    output[outputIndex] = normalized;
    if (!Number.isFinite(output[outputIndex])) {
      throw new ImagePreprocessingError(
        ImagePreprocessingErrorCode.NON_FINITE_OUTPUT,
      );
    }
  }
}

function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction !== 0.5) return Math.round(value);
  return floor % 2 === 0 ? floor : floor + 1;
}
