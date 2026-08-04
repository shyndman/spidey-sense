const IMAGE_MIME_TYPE_PATTERN = /^image\/[!#$%&'*+\-.^_`|~0-9a-z]+$/;
const SVG_MIME_TYPE = "image/svg+xml";
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const SVG_ANIMATION_ELEMENTS: Readonly<Record<string, true>> = {
  animate: true,
  animatecolor: true,
  animatemotion: true,
  animatetransform: true,
  discard: true,
  set: true,
};
const CSS_ANIMATION_DECLARATION_PATTERN =
  /(?:^|[;{])\s*(?:-[a-z]+-)?animation(?:-[a-z-]+)?\s*:/i;

export type ImageMimeType = `image/${string}`;

export type ImageResponseSelection =
  | { readonly kind: "eligible"; readonly mimeType: ImageMimeType }
  | { readonly kind: "ineligible" };

export const ImageDecodingErrorCode = {
  ANIMATED_IMAGE: "ANIMATED_IMAGE",
  DECODE_FAILED: "DECODE_FAILED",
  RASTERIZATION_FAILED: "RASTERIZATION_FAILED",
  UNSUPPORTED_TYPE: "UNSUPPORTED_TYPE",
} as const;

export type ImageDecodingErrorCode =
  (typeof ImageDecodingErrorCode)[keyof typeof ImageDecodingErrorCode];

const IMAGE_DECODING_ERROR_MESSAGES: Readonly<
  Record<ImageDecodingErrorCode, string>
> = {
  ANIMATED_IMAGE: "Animated image responses are not eligible for classification",
  DECODE_FAILED: "Image response could not be decoded",
  RASTERIZATION_FAILED: "Decoded image could not be converted to RGBA pixels",
  UNSUPPORTED_TYPE: "Firefox cannot safely decode this image response type",
};

/** A stable, content-free failure at the encoded-image boundary. */
export class ImageDecodingError extends Error {
  readonly code: ImageDecodingErrorCode;

  constructor(code: ImageDecodingErrorCode, cause?: unknown) {
    super(
      IMAGE_DECODING_ERROR_MESSAGES[code],
      cause === undefined ? undefined : { cause },
    );
    this.name = "ImageDecodingError";
    this.code = code;
  }
}

/**
 * Unpremultiplied RGBA8 pixels in the sRGB color space. The byte order is
 * red, green, blue, alpha for every pixel, row-major from the top-left. This is
 * the only decoded-image representation exposed to tensor preprocessing, so
 * later stages do not depend on DOM image objects or browser decoder state.
 */
export interface DecodedImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray<ArrayBuffer>;
  readonly channelOrder: "RGBA";
  readonly colorSpace: "srgb";
  readonly alphaMode: "unpremultiplied";
}

interface DecodedFrame {
  readonly source: CanvasImageSource;
  readonly width: number;
  readonly height: number;
  close(): void;
}

/**
 * Selects every response whose declared MIME essence is `image/*`. HTTP method
 * and status are deliberately absent: the product contract classifies any
 * image response and trusts Content-Type rather than sniffing encoded bytes.
 */
export function selectImageResponse(
  contentType: string | null,
): ImageResponseSelection {
  const mimeType = contentType?.split(";", 1)[0]?.trim().toLowerCase();
  if (mimeType === undefined || !IMAGE_MIME_TYPE_PATTERN.test(mimeType)) {
    console.debug("Response bypassed because Content-Type is not an image MIME type");
    return { kind: "ineligible" };
  }

  console.debug("Image response selected for in-memory decoding");
  return { kind: "eligible", mimeType: mimeType as ImageMimeType };
}

/**
 * Decodes one complete response entirely in memory and immediately rasterizes
 * it to the pipeline's RGBA8 sRGB boundary. Firefox's native ImageDecoder is
 * used whenever it supports the declared MIME type because its track metadata
 * can reject animations before a frame reaches preprocessing. Static SVG uses
 * Firefox's detached image-element decoder after an animation markup check.
 * Logs expose only MIME type, byte count, decoded dimensions, elapsed time,
 * and stable failure codes—never encoded or decoded content.
 */
export async function decodeImage(
  bytes: Uint8Array<ArrayBuffer>,
  mimeType: ImageMimeType,
): Promise<DecodedImage> {
  const startedAt = performance.now();
  let frame: DecodedFrame | undefined;
  try {
    frame = await decodeFrame(bytes, mimeType);
    const decoded = rasterizeFrame(frame);
    console.debug("Image response decoded into the in-memory pixel boundary", {
      durationMilliseconds: performance.now() - startedAt,
      encodedBytes: bytes.byteLength,
      mimeType,
      width: decoded.width,
      height: decoded.height,
    });
    return decoded;
  } catch (cause: unknown) {
    const error =
      cause instanceof ImageDecodingError
        ? cause
        : new ImageDecodingError(ImageDecodingErrorCode.DECODE_FAILED, cause);
    console.error(`Image decoding stopped: ${error.code}`, {
      durationMilliseconds: performance.now() - startedAt,
      encodedBytes: bytes.byteLength,
      mimeType,
    });
    throw error;
  } finally {
    frame?.close();
  }
}

async function decodeFrame(
  bytes: Uint8Array<ArrayBuffer>,
  mimeType: ImageMimeType,
): Promise<DecodedFrame> {
  if (await ImageDecoder.isTypeSupported(mimeType)) {
    return decodeTrackedFrame(bytes, mimeType);
  }
  if (mimeType === SVG_MIME_TYPE) return decodeSvgFrame(bytes);
  throw new ImageDecodingError(ImageDecodingErrorCode.UNSUPPORTED_TYPE);
}

async function decodeTrackedFrame(
  bytes: Uint8Array<ArrayBuffer>,
  mimeType: ImageMimeType,
): Promise<DecodedFrame> {
  const decoder = new ImageDecoder({
    colorSpaceConversion: "default",
    data: bytes,
    preferAnimation: true,
    type: mimeType,
  });

  try {
    await decoder.tracks.ready;
    for (let index = 0; index < decoder.tracks.length; index += 1) {
      if (decoder.tracks[index]?.animated === true) {
        throw new ImageDecodingError(
          ImageDecodingErrorCode.ANIMATED_IMAGE,
        );
      }
    }

    const result = await decoder.decode({
      completeFramesOnly: true,
      frameIndex: 0,
    });
    if (!result.complete) {
      result.image.close();
      throw new ImageDecodingError(ImageDecodingErrorCode.DECODE_FAILED);
    }

    return {
      source: result.image,
      width: result.image.displayWidth,
      height: result.image.displayHeight,
      close: () => {
        result.image.close();
        decoder.close();
      },
    };
  } catch (cause: unknown) {
    decoder.close();
    throw cause;
  }
}

async function decodeSvgFrame(
  bytes: Uint8Array<ArrayBuffer>,
): Promise<DecodedFrame> {
  if (svgContainsAnimation(bytes)) {
    throw new ImageDecodingError(ImageDecodingErrorCode.ANIMATED_IMAGE);
  }

  const objectUrl = URL.createObjectURL(
    new Blob([bytes], { type: SVG_MIME_TYPE }),
  );
  const image = document.createElement("img");
  image.decoding = "sync";
  image.src = objectUrl;

  const close = (): void => {
    image.src = "";
    URL.revokeObjectURL(objectUrl);
  };

  try {
    //! HACK: Firefox's ImageDecoder deliberately excludes SVG, while
    //! createImageBitmap rejects SVG blobs even though the browser image
    //! decoder supports them. A detached image element reaches that native
    //! decoder without entering the DOM or displaying content. The object URL
    //! exists only until synchronous rasterization finishes and is always
    //! revoked by the caller's resource cleanup.
    await image.decode();
    return {
      source: image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      close,
    };
  } catch (cause: unknown) {
    close();
    throw cause;
  }
}

function svgContainsAnimation(bytes: Uint8Array<ArrayBuffer>): boolean {
  const source = new TextDecoder().decode(bytes);
  const document = new DOMParser().parseFromString(source, SVG_MIME_TYPE);
  if (document.querySelector("parsererror") !== null) return false;

  for (const element of document.querySelectorAll("*")) {
    if (
      element.namespaceURI === SVG_NAMESPACE &&
      SVG_ANIMATION_ELEMENTS[element.localName.toLowerCase()] === true
    ) {
      return true;
    }
    const inlineStyle = element.getAttribute("style");
    if (
      inlineStyle !== null &&
      CSS_ANIMATION_DECLARATION_PATTERN.test(inlineStyle)
    ) {
      return true;
    }
  }

  for (const style of document.querySelectorAll("style")) {
    if (CSS_ANIMATION_DECLARATION_PATTERN.test(style.textContent ?? "")) {
      return true;
    }
  }
  return false;
}

function rasterizeFrame(frame: DecodedFrame): DecodedImage {
  try {
    if (frame.width <= 0 || frame.height <= 0) {
      throw new Error("Decoded frame dimensions must be positive");
    }
    const canvas = new OffscreenCanvas(frame.width, frame.height);
    const context = canvas.getContext("2d", {
      alpha: true,
      colorSpace: "srgb",
      willReadFrequently: true,
    });
    if (context === null) {
      throw new Error("Firefox did not provide a 2D canvas context");
    }
    context.drawImage(frame.source, 0, 0);
    const pixels = context.getImageData(0, 0, frame.width, frame.height, {
      colorSpace: "srgb",
    });
    return {
      width: frame.width,
      height: frame.height,
      data: pixels.data,
      channelOrder: "RGBA",
      colorSpace: "srgb",
      alphaMode: "unpremultiplied",
    };
  } catch (cause: unknown) {
    throw new ImageDecodingError(
      ImageDecodingErrorCode.RASTERIZATION_FAILED,
      cause,
    );
  }
}
