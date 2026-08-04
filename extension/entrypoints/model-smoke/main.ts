import {
  type DecodedImage,
  decodeImage,
  ImageDecodingError,
  ImageDecodingErrorCode,
  type ImageMimeType,
  selectImageResponse,
} from "../../src/image/decoding";
import { runModelInference } from "../../src/model/inference";
import { ModelSessionManager } from "../../src/model/session";

const PROBABILITY_SUM_TOLERANCE = 1e-5;
const PROXY_WIDTH = 2;
const PROXY_HEIGHT = 1;
const FULL_CHANNEL = 255;
const EMPTY_CHANNEL = 0;
const PROXY_RGBA = new Uint8ClampedArray([
  FULL_CHANNEL,
  EMPTY_CHANNEL,
  FULL_CHANNEL,
  FULL_CHANNEL,
  EMPTY_CHANNEL,
  FULL_CHANNEL,
  EMPTY_CHANNEL,
  FULL_CHANNEL,
]);
const STATIC_SVG_PROXY =
  '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">' +
  '<rect width="1" height="1" fill="#ff00ff"/>' +
  '<rect x="1" width="1" height="1" fill="#00ff00"/></svg>';
const ANIMATED_SVG_PROXY =
  '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">' +
  '<rect width="1" height="1" fill="#ff00ff">' +
  '<animate attributeName="fill" values="#ff00ff;#00ff00" dur="1s"/>' +
  "</rect></svg>";

function requireEligibleImage(contentType: string): ImageMimeType {
  const selection = selectImageResponse(contentType);
  if (selection.kind === "ineligible") {
    throw new Error("Harmless proxy MIME type was not selected");
  }
  return selection.mimeType;
}

async function createProxyPngBytes(): Promise<Uint8Array<ArrayBuffer>> {
  const canvas = new OffscreenCanvas(PROXY_WIDTH, PROXY_HEIGHT);
  const context = canvas.getContext("2d", { colorSpace: "srgb" });
  if (context === null) {
    throw new Error("Firefox did not provide a smoke-test canvas context");
  }
  context.putImageData(
    new ImageData(PROXY_RGBA, PROXY_WIDTH, PROXY_HEIGHT, {
      colorSpace: "srgb",
    }),
    0,
    0,
  );
  const blob = await canvas.convertToBlob({ type: "image/png" });
  return new Uint8Array(await blob.arrayBuffer());
}

function assertProxyPixels(decoded: DecodedImage): void {
  if (
    decoded.width !== PROXY_WIDTH ||
    decoded.height !== PROXY_HEIGHT ||
    decoded.channelOrder !== "RGBA" ||
    decoded.colorSpace !== "srgb" ||
    decoded.alphaMode !== "unpremultiplied" ||
    decoded.data.length !== PROXY_RGBA.length
  ) {
    throw new Error("Decoded proxy does not match the typed pixel boundary");
  }
  for (let index = 0; index < PROXY_RGBA.length; index += 1) {
    if (decoded.data[index] !== PROXY_RGBA[index]) {
      throw new Error("Decoded proxy pixels do not match their source");
    }
  }
}

/**
 * Creates harmless solid-color proxy bytes only in browser memory, decodes both
 * native raster and static SVG paths, and proves animation and non-image
 * responses stop before tensor preprocessing. Neither source nor decoded pixels
 * enter the DOM, logs, browser network, or filesystem.
 */
async function runFirefoxImageDecodeSmoke(): Promise<void> {
  const png = await decodeImage(
    await createProxyPngBytes(),
    requireEligibleImage("image/png"),
  );
  assertProxyPixels(png);

  const svg = await decodeImage(
    new TextEncoder().encode(STATIC_SVG_PROXY),
    requireEligibleImage("image/svg+xml"),
  );
  assertProxyPixels(svg);

  if (selectImageResponse("application/octet-stream").kind !== "ineligible") {
    throw new Error("Non-image response entered the decoding pipeline");
  }

  try {
    await decodeImage(
      new TextEncoder().encode(ANIMATED_SVG_PROXY),
      requireEligibleImage("image/svg+xml"),
    );
  } catch (cause: unknown) {
    if (
      cause instanceof ImageDecodingError &&
      cause.code === ImageDecodingErrorCode.ANIMATED_IMAGE
    ) {
      return;
    }
    throw cause;
  }
  throw new Error("Animated image entered the decoding pipeline");
}

const SmokeStatus = {
  PASSED: "passed",
  FAILED: "failed",
} as const;

type SmokeStatus = (typeof SmokeStatus)[keyof typeof SmokeStatus];

/**
 * Exercises the exact packaged metadata, model, and ONNX Runtime assets from a
 * real Firefox extension context. The numeric input is an all-zero synthetic
 * tensor—not an encoded, decoded, or preprocessed image—so this page can prove
 * runtime initialization and inference without acquiring or persisting imagery.
 * The page is unlisted and has no manifest or product UI route; WebDriver opens
 * its internal extension URL directly during the explicit smoke command.
 */
async function runFirefoxModelSmoke(): Promise<void> {
  const metadataUrl = new URL(
    import.meta.env.WXT_MODEL_METADATA_PATH,
    globalThis.location.href,
  );
  const manager = new ModelSessionManager(metadataUrl);
  const { metadata, session } = await manager.initialize();
  const [, channels, height, width] = metadata.input.shape;
  const syntheticInput = new Float32Array(channels * height * width);
  const result = await runModelInference(session, metadata, syntheticInput);

  if (result.probabilities.length !== metadata.output.shape[1]) {
    throw new Error("Smoke output length does not match generated metadata");
  }

  let probabilitySum = 0;
  for (const probability of result.probabilities) {
    if (!Number.isFinite(probability)) {
      throw new Error("Smoke output contains a non-finite probability");
    }
    probabilitySum += probability;
  }
  if (Math.abs(probabilitySum - 1) > PROBABILITY_SUM_TOLERANCE) {
    throw new Error("Smoke output probabilities are not normalized");
  }
}

async function runFirefoxExtensionSmoke(): Promise<void> {
  await runFirefoxImageDecodeSmoke();
  await runFirefoxModelSmoke();
}

function publishStatus(status: SmokeStatus): void {
  const result = document.querySelector<HTMLElement>("#smoke-result");
  if (result === null) {
    throw new Error("Smoke result element is missing");
  }
  result.dataset.status = status;
  result.textContent = status;
}

void runFirefoxExtensionSmoke().then(
  () => publishStatus(SmokeStatus.PASSED),
  () => {
    console.error("Firefox extension smoke test failed");
    publishStatus(SmokeStatus.FAILED);
  },
);
