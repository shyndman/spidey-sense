import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  type InterceptProofImageMimeType,
  INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE,
} from './intercept-proof/image-bytes';
import type { ImageReplacementWebRequest } from './image-replacement';
import { registerImageReplacement } from './image-replacement';

type RequestListener = Parameters<
  ImageReplacementWebRequest['onHeadersReceived']['addListener']
>[0];
type RequestDetails = Parameters<RequestListener>[0];
type RequestFilter = Parameters<
  ImageReplacementWebRequest['onHeadersReceived']['addListener']
>[1];
type RequestExtraInfoSpec = Parameters<
  ImageReplacementWebRequest['onHeadersReceived']['addListener']
>[2];
type ResponseHeader = NonNullable<RequestDetails['responseHeaders']>[number];
type StreamFilter = ReturnType<ImageReplacementWebRequest['filterResponseData']>;

const IMAGE_REPLACEMENTS = [
  { mimeType: 'image/png' },
  { mimeType: 'image/jpeg' },
  { mimeType: 'image/gif' },
  { mimeType: 'image/webp' },
  { mimeType: 'image/svg+xml' },
] as const satisfies ReadonlyArray<{
  mimeType: InterceptProofImageMimeType;
}>;

function responseHeader(name: string, value?: string): ResponseHeader {
  return value === undefined ? { name } : { name, value };
}

function requestDetails(
  overrides: Partial<RequestDetails> = {},
): RequestDetails {
  return {
    requestId: 'request-id',
    url: 'https://example.com/image.png',
    responseHeaders: [responseHeader('Content-Type', 'image/png')],
    ...overrides,
  };
}

function createFilter(error = 'Unexpected filter error') {
  return {
    error,
    onstart: null as StreamFilter['onstart'],
    onerror: null as StreamFilter['onerror'],
    write: vi.fn<StreamFilter['write']>(),
    close: vi.fn<StreamFilter['close']>(),
  };
}

function installReplacement(filter = createFilter()) {
  let registeredListener: RequestListener | undefined;
  let registeredFilter: RequestFilter | undefined;
  let registeredExtraInfoSpec: RequestExtraInfoSpec | undefined;
  const addListener = vi.fn<
    ImageReplacementWebRequest['onHeadersReceived']['addListener']
  >((listener, requestFilter, extraInfoSpec) => {
    registeredListener = listener;
    registeredFilter = requestFilter;
    registeredExtraInfoSpec = extraInfoSpec;
  });
  const filterResponseData = vi.fn<
    ImageReplacementWebRequest['filterResponseData']
  >(() => filter);

  registerImageReplacement({
    onHeadersReceived: { addListener },
    filterResponseData,
  });

  if (
    registeredListener === undefined ||
    registeredFilter === undefined ||
    registeredExtraInfoSpec === undefined
  ) {
    throw new Error('Expected image replacement listener registration');
  }

  return {
    addListener,
    filter,
    filterResponseData,
    listener: registeredListener,
    registrationFilter: registeredFilter,
    registrationExtraInfoSpec: registeredExtraInfoSpec,
  };
}

function expectFailureContext(
  calls: ReadonlyArray<ReadonlyArray<unknown>>,
  context: Readonly<Record<string, string>>,
): void {
  const contextualCall = calls.find((call) =>
    call.some((argument) => {
      if (typeof argument !== 'object' || argument === null) return false;
      const record = argument as Record<string, unknown>;
      return Object.entries(context).every(
        ([key, value]) => record[key] === value,
      );
    }),
  );
  expect(contextualCall).toBeDefined();
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('registerImageReplacement', () => {
  it('registers only image requests for the five approved destination hosts', () => {
    const { addListener } = installReplacement();

    expect(addListener).toHaveBeenCalledExactlyOnceWith(
      expect.any(Function),
      {
        urls: [
          'https://yt3.ggpht.com/*',
          'https://i.ytimg.com/*',
          'https://external-preview.redd.it/*',
          'https://preview.redd.it/*',
          'https://i.redd.it/*',
        ],
        types: ['image', 'imageset'],
      },
      ['blocking', 'responseHeaders'],
    );
    expect(addListener.mock.calls[0]).toHaveLength(3);
  });

  it.each(IMAGE_REPLACEMENTS)(
    'writes the exact $mimeType replacement bytes and closes at stream start',
    ({ mimeType }) => {
      const { filter, listener } = installReplacement();
      const bytes = INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE[mimeType];

      const result = listener(
        requestDetails({
          url: 'https://example.com/image.bin',
          responseHeaders: [
            responseHeader(
              'Content-Type',
              `${mimeType.toUpperCase()}; charset=UTF-8`,
            ),
          ],
        }),
      );
      expect(filter.write).not.toHaveBeenCalled();
      filter.onstart?.();

      expect(result).toEqual({
        responseHeaders: [
          responseHeader('Content-Type', mimeType),
          responseHeader('Content-Length', String(bytes.byteLength)),
        ],
      });
      expect(filter.write).toHaveBeenCalledExactlyOnceWith(bytes);
      expect(filter.close).toHaveBeenCalledExactlyOnceWith();
    },
  );

  it('normalizes MIME case and parameters before selecting replacement bytes', () => {
    const { filter, listener } = installReplacement();
    const bytes = INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE['image/svg+xml'];

    listener(
      requestDetails({
        url: 'https://example.com/image.bin',
        responseHeaders: [
          responseHeader('Content-Type', '  ImAgE/SvG+XmL ; charset=utf-8  '),
        ],
      }),
    );
    filter.onstart?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(bytes);
  });

  it.each([
    ['.jpg', 'image/jpeg'],
    ['.jpeg', 'image/jpeg'],
    ['.PNG', 'image/png'],
    ['.gif', 'image/gif'],
    ['.webp', 'image/webp'],
    ['.SVG', 'image/svg+xml'],
  ] as const)(
    'falls back to case-insensitive pathname extension %s while ignoring query',
    (extension, mimeType) => {
      const { filter, listener } = installReplacement();
      const url = `https://example.com/path/photo${extension}?size=large`;

      listener(
        requestDetails({
          url,
          responseHeaders: [responseHeader('X-Source', 'fallback')],
        }),
      );
      filter.onstart?.();

      expect(filter.write).toHaveBeenCalledExactlyOnceWith(
        INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE[mimeType],
      );
    },
  );

  it('uses a supported Content-Type instead of a conflicting URL extension', () => {
    const { filter, listener } = installReplacement();

    listener(
      requestDetails({
        url: 'https://example.com/photo.jpg',
        responseHeaders: [
          responseHeader('Content-Type', 'image/png; charset=binary'),
        ],
      }),
    );
    filter.onstart?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(
      INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE['image/png'],
    );
  });

  it('falls back to the URL extension when Content-Type is unsupported', () => {
    const { filter, listener } = installReplacement();

    listener(
      requestDetails({
        url: 'https://example.com/photo.JPEG?download=1',
        responseHeaders: [
          responseHeader('Content-Type', 'application/octet-stream'),
        ],
      }),
    );
    filter.onstart?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(
      INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE['image/jpeg'],
    );
  });

  it('logs each intercepted URL with its raw Content-Type before setup', () => {
    const infoLog = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const { listener } = installReplacement();
    const url = 'https://example.com/setup.png?source=test';
    const rawContentType = 'ImAgE/PNG; Charset=UTF-8';

    listener(
      requestDetails({
        url,
        responseHeaders: [responseHeader('Content-Type', rawContentType)],
      }),
    );

    expect(infoLog).toHaveBeenCalledExactlyOnceWith(
      'Intercepting image URL',
      url,
      rawContentType,
    );
  });

  it('rewrites replacement headers while preserving unrelated headers', () => {
    const { listener } = installReplacement();
    const bytes = INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE['image/png'];
    const url = 'https://example.com/photo.jpg';
    const result = listener(
      requestDetails({
        url,
        responseHeaders: [
          responseHeader('X-Trace', 'trace-id'),
          responseHeader('cOnTeNt-TyPe', 'IMAGE/PNG'),
          responseHeader('Content-Length', '999'),
          responseHeader('CONTENT-ENCODING', 'gzip'),
          responseHeader('X-After', 'preserve'),
        ],
      }),
    );

    expect(result).toEqual({
      responseHeaders: [
        responseHeader('X-Trace', 'trace-id'),
        responseHeader('X-After', 'preserve'),
        responseHeader('Content-Type', 'image/png'),
        responseHeader('Content-Length', String(bytes.byteLength)),
      ],
    });
  });

  it('cancels unsupported signals before filter creation and logs all context', () => {
    const infoLog = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filterResponseData, listener } = installReplacement();
    const url = 'https://example.com/file.bin?download=1';
    const rawContentType = 'application/octet-stream';

    const result = listener(
      requestDetails({
        url,
        responseHeaders: [responseHeader('Content-Type', rawContentType)],
      }),
    );

    expect(result).toEqual({ cancel: true });
    expect(filterResponseData).not.toHaveBeenCalled();
    expect(infoLog).toHaveBeenCalledExactlyOnceWith(
      'Intercepting image URL',
      url,
      rawContentType,
    );
    expectFailureContext(errorLog.mock.calls, {
      url,
      contentType: rawContentType,
      extension: '.bin',
      reason: 'No supported response MIME type or pathname extension',
    });
  });

  it('cancels missing signals before filter creation and logs missing context', () => {
    const infoLog = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filterResponseData, listener } = installReplacement();
    const url = 'https://example.com/file';

    const result = listener(
      requestDetails({
        url,
        responseHeaders: [responseHeader('X-Source', 'missing')],
      }),
    );

    expect(result).toEqual({ cancel: true });
    expect(filterResponseData).not.toHaveBeenCalled();
    expect(infoLog).toHaveBeenCalledExactlyOnceWith(
      'Intercepting image URL',
      url,
      '<missing>',
    );
    expectFailureContext(errorLog.mock.calls, {
      url,
      contentType: '<missing>',
      extension: '<missing>',
      reason: 'No supported response MIME type or pathname extension',
    });
  });

  it('cancels and logs URL/type context when filter setup fails', () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filterResponseData, listener } = installReplacement();
    const url = 'https://example.com/setup-failure.png';
    const rawContentType = 'image/png; charset=UTF-8';
    filterResponseData.mockImplementationOnce(() => {
      throw new Error('filter setup failed');
    });

    const result = listener(
      requestDetails({
        url,
        responseHeaders: [responseHeader('Content-Type', rawContentType)],
      }),
    );

    expect(result).toEqual({ cancel: true });
    expect(filterResponseData).toHaveBeenCalledExactlyOnceWith('request-id');
    expectFailureContext(errorLog.mock.calls, {
      url,
      contentType: rawContentType,
      extension: '.png',
      reason: 'Filter setup failed',
    });
  });

  it('logs stream errors without writing or closing the replacement filter', () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filter, listener } = installReplacement();
    const url = 'https://example.com/stream-error.png';
    const rawContentType = 'image/png';

    listener(
      requestDetails({
        url,
        responseHeaders: [responseHeader('Content-Type', rawContentType)],
      }),
    );
    filter.onerror?.();
    filter.onstart?.();

    expect(filter.write).not.toHaveBeenCalled();
    expect(filter.close).not.toHaveBeenCalled();
    expectFailureContext(errorLog.mock.calls, {
      url,
      contentType: rawContentType,
      extension: '.png',
      reason: 'Unexpected filter error',
    });
  });

  it('logs write failures and closes once after a failed write', () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filter, listener } = installReplacement();
    const url = 'https://example.com/write-failure.png';
    filter.write.mockImplementationOnce(() => {
      throw new Error('write failed');
    });

    listener(requestDetails({ url }));
    filter.onstart?.();
    filter.onstart?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(
      INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE['image/png'],
    );
    expect(filter.close).toHaveBeenCalledExactlyOnceWith();
    expectFailureContext(errorLog.mock.calls, {
      url,
      contentType: 'image/png',
      extension: '.png',
      reason: 'Writing replacement bytes failed',
    });
  });

  it('logs close failures after one replacement write and one close attempt', () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filter, listener } = installReplacement();
    const url = 'https://example.com/close-failure.png';
    filter.close.mockImplementationOnce(() => {
      throw new Error('close failed');
    });

    listener(requestDetails({ url }));
    filter.onstart?.();
    filter.onstart?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(
      INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE['image/png'],
    );
    expect(filter.close).toHaveBeenCalledExactlyOnceWith();
    expectFailureContext(errorLog.mock.calls, {
      url,
      contentType: 'image/png',
      extension: '.png',
      reason: 'Closing replacement stream failed',
    });
  });
});
