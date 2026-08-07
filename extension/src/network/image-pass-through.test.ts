import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ImagePassThroughWebRequest } from './image-pass-through';
import { registerImagePassThrough } from './image-pass-through';

type RequestListener = Parameters<
  ImagePassThroughWebRequest['onBeforeRequest']['addListener']
>[0];
type RequestFilter = Parameters<
  ImagePassThroughWebRequest['onBeforeRequest']['addListener']
>[1];
type RequestExtraInfoSpec = Parameters<
  ImagePassThroughWebRequest['onBeforeRequest']['addListener']
>[2];

function arrayBuffer(...bytes: number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.length);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

function installPassThrough(error = 'Unexpected filter error') {
  const filter = {
    error,
    ondata: null as
      | ((event: { readonly data: ArrayBuffer }) => void)
      | null,
    onerror: null as (() => void) | null,
    onstop: null as (() => void) | null,
    write: vi.fn<(data: Uint8Array<ArrayBuffer>) => void>(),
    close: vi.fn<() => void>(),
  };

  let registeredListener: RequestListener | undefined;
  let registeredFilter: RequestFilter | undefined;
  let registeredExtraInfoSpec: RequestExtraInfoSpec | undefined;
  const addListener = vi.fn<
    ImagePassThroughWebRequest['onBeforeRequest']['addListener']
  >((listener, requestFilter, extraInfoSpec) => {
    registeredListener = listener;
    registeredFilter = requestFilter;
    registeredExtraInfoSpec = extraInfoSpec;
  });
  const filterResponseData = vi.fn<
    ImagePassThroughWebRequest['filterResponseData']
  >(() => filter);

  registerImagePassThrough({
    onBeforeRequest: { addListener },
    filterResponseData,
  });

  if (
    registeredListener === undefined ||
    registeredFilter === undefined ||
    registeredExtraInfoSpec === undefined
  ) {
    throw new Error('Expected pass-through listener registration');
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

afterEach(() => {
  vi.restoreAllMocks();
});

describe('registerImagePassThrough', () => {
  it('registers only image requests for the three approved destination hosts', () => {
    const { addListener } = installPassThrough();

    expect(addListener).toHaveBeenCalledExactlyOnceWith(
      expect.any(Function),
      {
        urls: [
          'https://yt3.ggpht.com/*',
          'https://preview.redd.it/*',
          'https://i.redd.it/*',
        ],
        types: ['image'],
      },
      ['blocking'],
    );
    expect(addListener.mock.calls[0]).toHaveLength(3);
  });

  it('buffers chunks until stop, then writes the ordered body once and closes once', () => {
    const { filter, filterResponseData, listener } = installPassThrough();
    listener({
      requestId: 'request-id',
      url: 'https://example.com/image.png',
    });

    filter.ondata?.({ data: arrayBuffer(1, 2) });
    filter.ondata?.({ data: arrayBuffer(3, 4, 5) });
    expect(filterResponseData).toHaveBeenCalledExactlyOnceWith('request-id');
    expect(filter.write).not.toHaveBeenCalled();

    filter.onstop?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(
      new Uint8Array([1, 2, 3, 4, 5]),
    );
    expect(filter.close).toHaveBeenCalledExactlyOnceWith();
  });

  it('writes and closes an empty response exactly once', () => {
    const { filter, listener } = installPassThrough();
    listener({
      requestId: 'empty-request-id',
      url: 'https://example.com/empty.png',
    });

    filter.onstop?.();

    expect(filter.write).toHaveBeenCalledExactlyOnceWith(new Uint8Array(0));
    expect(filter.close).toHaveBeenCalledExactlyOnceWith();
  });

  it('logs each intercepted image URL exactly once before filter setup', () => {
    const infoLog = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const { filterResponseData, listener } = installPassThrough();
    const imageUrl = 'https://example.com/setup-failure.png';
    filterResponseData.mockImplementationOnce(() => {
      throw new Error('filter setup failed');
    });

    listener({
      requestId: 'setup-failure-request-id',
      url: imageUrl,
    });

    expect(infoLog).toHaveBeenCalledExactlyOnceWith(
      'Intercepting image URL',
      imageUrl,
    );
  });

  it('quietly terminates the old filter after service-worker fallback redirection', () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filter, filterResponseData, listener } = installPassThrough(
      'ServiceWorker fallback redirection',
    );
    listener({
      requestId: 'fallback-request-id',
      url: 'https://example.com/fallback.png',
    });

    filter.ondata?.({ data: arrayBuffer(9, 8, 7) });
    filter.onerror?.();
    filter.ondata?.({ data: arrayBuffer(6, 5, 4) });
    filter.onstop?.();

    expect(errorLog).not.toHaveBeenCalled();
    expect(filterResponseData).toHaveBeenCalledExactlyOnceWith(
      'fallback-request-id',
    );
    expect(filter.write).not.toHaveBeenCalled();
    expect(filter.close).not.toHaveBeenCalled();
  });

  it('logs the specific reason when the filter fails', () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const { filter, listener } = installPassThrough('NetworkError');
    listener({
      requestId: 'private-request-id',
      url: 'https://example.com/private.png',
    });

    filter.ondata?.({ data: arrayBuffer(9, 8, 7) });
    filter.onerror?.();

    expect(errorLog).toHaveBeenCalledExactlyOnceWith(
      'Image response pass-through failed',
      'NetworkError',
    );
    expect(filter.write).not.toHaveBeenCalled();
    expect(filter.close).not.toHaveBeenCalled();
  });
});
