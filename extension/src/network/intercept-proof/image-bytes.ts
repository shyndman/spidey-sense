import gifBytes from './images/magenta.gif?bytes';
import jpegBytes from './images/magenta.jpg?bytes';
import pngBytes from './images/magenta.png?bytes';
import svgBytes from './images/magenta.svg?bytes';
import webpBytes from './images/magenta.webp?bytes';

export type InterceptProofImageMimeType =
  | 'image/png'
  | 'image/jpeg'
  | 'image/gif'
  | 'image/webp'
  | 'image/svg+xml';

/**
 * Exact format-matched response bodies for interception proof replacements.
 *
 * The synchronous imports keep each harmless proxy available at module
 * initialization, so a blocked image response can be replaced immediately
 * with bytes whose body format matches its declared MIME type.
 */
export const INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE: Readonly<
  Record<InterceptProofImageMimeType, Uint8Array<ArrayBuffer>>
> = {
  'image/png': pngBytes,
  'image/jpeg': jpegBytes,
  'image/gif': gifBytes,
  'image/webp': webpBytes,
  'image/svg+xml': svgBytes,
};
