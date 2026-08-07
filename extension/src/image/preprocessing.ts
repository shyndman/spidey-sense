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
  readonly data: Uint8ClampedArray;
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
  readonly targetWidth: number;
  readonly targetHeight: number;
  readonly paddingLeft: number;
  readonly paddingTop: number;
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
 * input described by generated model metadata. The complete source image is
 * fitted inside the model input while preserving its aspect ratio, upscaling
 * small inputs, and centering any unused area on black padding. Premultiplied-
 * alpha triangle resizing prevents hidden colors in transparent pixels from
 * bleeding into neighbors before pixels are composited against black.
 *
 * Transformation is fail-closed at this boundary: every invalid image,
 * malformed transform, resize failure, and non-finite output throws a typed
 * error. Logs expose only elapsed time, dimensions, element counts, and stable
 * failure codes—never pixels, tensor values, metadata values, or URLs.
 */
export async function transformImageToModelInput(
  image: DecodedImage,
  metadata: ModelMetadata,
  dependencies: ImagePreprocessingDependencies = defaultDependencies,
): Promise<Float32Array> {
  const startedAt = performance.now();
  try {
    validateImage(image);
    validateNormalization(metadata);
    const geometry = deriveContainGeometry(image, metadata);
    const resized = await dependencies.resize(
      image,
      geometry.width,
      geometry.height,
    );
    validateResizeOutput(resized, geometry);
    const output = normalizeContainedImage(resized, metadata, geometry);
    console.debug("Decoded image transformed into the model input boundary", {
      durationMilliseconds: performance.now() - startedAt,
      sourceWidth: image.width,
      sourceHeight: image.height,
      resizedWidth: geometry.width,
      resizedHeight: geometry.height,
      paddingLeft: geometry.paddingLeft,
      paddingTop: geometry.paddingTop,
      outputElements: output.length,
    });
    return output;
  } catch (cause: unknown) {
    const error =
      cause instanceof ImagePreprocessingError
        ? cause
        : new ImagePreprocessingError(
            ImagePreprocessingErrorCode.TRANSFORMATION_FAILED,
            cause,
          );
    console.error(`Image preprocessing stopped: ${error.code}`, {
      durationMilliseconds: performance.now() - startedAt,
      sourceWidth: image.width,
      sourceHeight: image.height,
    });
    throw error;
  }
}

function validateImage(image: DecodedImage): void {
  const imageMetadata: Readonly<{
    channelOrder: string;
    colorSpace: string;
    alphaMode: string;
  }> = image;
  if (
    !Number.isSafeInteger(image.width) ||
    image.width <= 0 ||
    !Number.isSafeInteger(image.height) ||
    image.height <= 0 ||
    imageMetadata.channelOrder !== "RGBA" ||
    imageMetadata.colorSpace !== "srgb" ||
    imageMetadata.alphaMode !== "unpremultiplied"
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

function deriveContainGeometry(
  image: DecodedImage,
  metadata: ModelMetadata,
): ResizeGeometry {
  const targetHeight = metadata.input.shape[2];
  const targetWidth = metadata.input.shape[3];
  const scale = Math.min(
    targetWidth / image.width,
    targetHeight / image.height,
  );
  const width = Math.min(
    targetWidth,
    Math.max(1, Math.round(image.width * scale)),
  );
  const height = Math.min(
    targetHeight,
    Math.max(1, Math.round(image.height * scale)),
  );

  return {
    width,
    height,
    targetWidth,
    targetHeight,
    paddingLeft: Math.floor((targetWidth - width) / 2),
    paddingTop: Math.floor((targetHeight - height) / 2),
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

function normalizeContainedImage(
  resized: ResizedRgbaImage,
  metadata: ModelMetadata,
  geometry: ResizeGeometry,
): Float32Array {
  const channelPlaneLength = geometry.targetWidth * geometry.targetHeight;
  const output = new Float32Array(
    RGB_CHANNEL_INDICES.length * channelPlaneLength,
  );

  for (const channel of RGB_CHANNEL_INDICES) {
    const black =
      (0 - metadata.input.mean[channel]) /
      metadata.input.standardDeviation[channel];
    if (!Number.isFinite(black)) {
      throw new ImagePreprocessingError(
        ImagePreprocessingErrorCode.NON_FINITE_OUTPUT,
      );
    }
    output.fill(
      black,
      channel * channelPlaneLength,
      (channel + 1) * channelPlaneLength,
    );
  }

  for (let y = 0; y < resized.height; y += 1) {
    for (let x = 0; x < resized.width; x += 1) {
      const sourceOffset = (y * resized.width + x) * RGBA_CHANNEL_COUNT;
      const outputOffset =
        (geometry.paddingTop + y) * geometry.targetWidth +
        geometry.paddingLeft +
        x;
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
  pixels: Uint8ClampedArray,
  sourceOffset: number,
  metadata: ModelMetadata,
): void {
  const alphaValue = pixels.at(sourceOffset + ALPHA_CHANNEL_INDEX);
  if (alphaValue === undefined) {
    throw new ImagePreprocessingError(
      ImagePreprocessingErrorCode.INVALID_RESIZE_OUTPUT,
    );
  }
  const alpha = alphaValue / MAX_CHANNEL_VALUE;
  for (const channel of RGB_CHANNEL_INDICES) {
    const channelValue = pixels.at(sourceOffset + channel);
    if (channelValue === undefined) {
      throw new ImagePreprocessingError(
        ImagePreprocessingErrorCode.INVALID_RESIZE_OUTPUT,
      );
    }
    const composited = channelValue * alpha;
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

