import { browser } from 'wxt/browser';

import {
  INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE,
  type InterceptProofImageMimeType,
} from './intercept-proof/image-bytes';
import { IMAGE_HOST_PATTERNS } from './image-host-patterns';

export interface ImageReplacementStreamFilter {
  readonly error: string;
  onstart: (() => void) | null;
  onerror: (() => void) | null;
  write(data: Uint8Array<ArrayBuffer>): void;
  close(): void;
}

export interface ImageReplacementResponseHeader {
  readonly name: string;
  readonly value?: string;
}

export interface ImageReplacementHeadersReceivedDetails {
  readonly requestId: string;
  readonly url: string;
  readonly responseHeaders?: readonly ImageReplacementResponseHeader[];
}

export type ImageReplacementBlockingResponse =
  | { readonly cancel: true }
  | { readonly responseHeaders: ImageReplacementResponseHeader[] };

export interface ImageReplacementWebRequest {
  readonly onHeadersReceived: {
    addListener(
      listener: (
        details: ImageReplacementHeadersReceivedDetails,
      ) => ImageReplacementBlockingResponse,
      filter: {
        readonly urls: readonly string[];
        readonly types: readonly ['image', 'imageset'];
      },
      extraInfoSpec: readonly ['blocking', 'responseHeaders'],
    ): void;
  };
  filterResponseData(requestId: string): ImageReplacementStreamFilter;
}

type ImageReplacementExtension =
  | '.png'
  | '.jpg'
  | '.jpeg'
  | '.gif'
  | '.webp'
  | '.svg';

interface ReplacementFailureContext {
  readonly url: string;
  readonly rawContentType?: string;
  readonly extension?: string;
  readonly reason: string;
}

const IMAGE_REPLACEMENT_EXTRA_INFO_SPEC = [
  'blocking',
  'responseHeaders',
] as const;
const IMAGE_RESOURCE_TYPES = ['image', 'imageset'] as const;
const IMAGE_REPLACEMENT_ERROR = 'Image response replacement failed';
const IMAGE_REPLACEMENT_LOG_MESSAGE = 'Intercepting image URL';
const MIME_TYPE_BY_EXTENSION: Readonly<
  Record<ImageReplacementExtension, InterceptProofImageMimeType>
> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
};

function isSupportedMimeType(
  value: string,
): value is InterceptProofImageMimeType {
  return Object.hasOwn(INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE, value);
}

function normalizeContentType(
  rawContentType: string | undefined,
): InterceptProofImageMimeType | undefined {
  const normalized = rawContentType?.split(';', 1)[0]?.trim().toLowerCase();
  return normalized !== undefined && isSupportedMimeType(normalized)
    ? normalized
    : undefined;
}

function getRawContentType(
  responseHeaders: readonly ImageReplacementResponseHeader[] | undefined,
): string | undefined {
  return responseHeaders?.find(
    ({ name }) => name.toLowerCase() === 'content-type',
  )?.value;
}

function getPathnameExtension(url: string): string | undefined {
  let pathname: string;
  try {
    pathname = new URL(url).pathname;
  } catch {
    return undefined;
  }

  return /\.([^./]+)$/.exec(pathname)?.[0].toLowerCase();
}

function isSupportedExtension(
  extension: string,
): extension is ImageReplacementExtension {
  return Object.hasOwn(MIME_TYPE_BY_EXTENSION, extension);
}

function getPathnameMimeType(
  extension: string | undefined,
): InterceptProofImageMimeType | undefined {
  return extension !== undefined && isSupportedExtension(extension)
    ? MIME_TYPE_BY_EXTENSION[extension]
    : undefined;
}

function getReplacementMimeType(
  rawContentType: string | undefined,
  extension: string | undefined,
): InterceptProofImageMimeType | undefined {
  return normalizeContentType(rawContentType) ?? getPathnameMimeType(extension);
}

function isReplacementHeader(name: string): boolean {
  const normalizedName = name.toLowerCase();
  return (
    normalizedName === 'content-type' ||
    normalizedName === 'content-length' ||
    normalizedName === 'content-encoding'
  );
}

function getReplacementResponseHeaders(
  responseHeaders: readonly ImageReplacementResponseHeader[] | undefined,
  mimeType: InterceptProofImageMimeType,
  byteLength: number,
): ImageReplacementResponseHeader[] {
  return [
    ...(responseHeaders?.filter(({ name }) => !isReplacementHeader(name)) ?? []),
    { name: 'Content-Type', value: mimeType },
    { name: 'Content-Length', value: String(byteLength) },
  ];
}

function logInterception(url: string, rawContentType: string | undefined): void {
  console.info(
    IMAGE_REPLACEMENT_LOG_MESSAGE,
    url,
    rawContentType ?? '<missing>',
  );
}

function logReplacementFailure(context: ReplacementFailureContext): void {
  console.error(IMAGE_REPLACEMENT_ERROR, {
    url: context.url,
    contentType: context.rawContentType ?? '<missing>',
    extension: context.extension ?? '<missing>',
    reason: context.reason,
  });
}

function configureReplacementFilter(
  filter: ImageReplacementStreamFilter,
  replacementBytes: Uint8Array<ArrayBuffer>,
  context: ReplacementFailureContext,
): void {
  let streamEnded = false;
  filter.onstart = () => {
    if (streamEnded) return;
    streamEnded = true;
    try {
      filter.write(replacementBytes);
    } catch {
      logReplacementFailure({
        ...context,
        reason: 'Writing replacement bytes failed',
      });
    }
    try {
      filter.close();
    } catch {
      logReplacementFailure({
        ...context,
        reason: 'Closing replacement stream failed',
      });
    }
  };
  filter.onerror = () => {
    if (streamEnded) return;
    streamEnded = true;
    logReplacementFailure({ ...context, reason: filter.error });
  };
}

/**
 * Registers blocking image interception for the configured image hosts.
 *
 * The replacement MIME type is selected from the normalized response
 * `Content-Type` first, then from a supported pathname extension. Supported
 * responses receive immediate 1x1 proxy bytes with matching
 * `Content-Type` and `Content-Length` headers. Interception fails closed by
 * cancelling responses with neither supported signal and responses whose
 * filter setup fails.
 */
export function registerImageReplacement(
  //! WXT types omit Firefox's permission-gated response stream API.
  webRequest: ImageReplacementWebRequest =
    browser.webRequest as unknown as ImageReplacementWebRequest,
): void {
  webRequest.onHeadersReceived.addListener(
    (details) => {
      const rawContentType = getRawContentType(details.responseHeaders);
      const extension = getPathnameExtension(details.url);
      logInterception(details.url, rawContentType);
      const mimeType = getReplacementMimeType(rawContentType, extension);
      const context = {
        url: details.url,
        rawContentType,
        extension,
        reason: 'No supported response MIME type or pathname extension',
      } satisfies ReplacementFailureContext;
      if (mimeType === undefined) {
        logReplacementFailure(context);
        return { cancel: true };
      }

      const replacementBytes =
        INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE[mimeType];
      let filter: ImageReplacementStreamFilter;
      try {
        filter = webRequest.filterResponseData(details.requestId);
      } catch {
        logReplacementFailure({
          ...context,
          reason: 'Filter setup failed',
        });
        return { cancel: true };
      }

      configureReplacementFilter(filter, replacementBytes, context);
      return {
        responseHeaders: getReplacementResponseHeaders(
          details.responseHeaders,
          mimeType,
          replacementBytes.byteLength,
        ),
      };
    },
    {
      urls: IMAGE_HOST_PATTERNS,
      types: IMAGE_RESOURCE_TYPES,
    },
    IMAGE_REPLACEMENT_EXTRA_INFO_SPEC,
  );
}
