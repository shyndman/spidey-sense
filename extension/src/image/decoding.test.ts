import {
  afterEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";

import {
  decodeImage,
  ImageDecodingErrorCode,
  selectImageResponse,
} from "./decoding";

interface BrowserStubOptions {
  readonly animated?: boolean;
  readonly decodeError?: Error;
  readonly supported?: boolean;
}

interface BrowserStubState {
  readonly closeDecoder: Mock<() => void>;
  readonly closeFrame: Mock<() => void>;
  readonly decode: Mock<() => Promise<ImageDecodeResult>>;
  readonly pixels: Uint8ClampedArray<ArrayBuffer>;
}

function installBrowserStubs(
  options: BrowserStubOptions = {},
): BrowserStubState {
  const animated = options.animated ?? false;
  const supported = options.supported ?? true;
  const closeDecoder = vi.fn<() => void>();
  const closeFrame = vi.fn<() => void>();
  const pixels = new Uint8ClampedArray([11, 22, 33, 44]);
  const frame = {
    close: closeFrame,
    displayHeight: 1,
    displayWidth: 1,
  } as unknown as VideoFrame;
  const decode = vi.fn<() => Promise<ImageDecodeResult>>(() => {
    if (options.decodeError !== undefined) {
      return Promise.reject(options.decodeError);
    }
    return Promise.resolve({ complete: true, image: frame });
  });
  const track: ImageTrack = {
    animated,
    frameCount: animated ? 2 : 1,
    repetitionCount: animated ? Number.POSITIVE_INFINITY : 0,
    selected: true,
  };
  const tracks: ImageTrackList = {
    0: track,
    length: 1,
    ready: Promise.resolve(),
    selectedIndex: 0,
    selectedTrack: track,
    [Symbol.iterator]: () => [track][Symbol.iterator](),
  };

  class StubImageDecoder {
    static isTypeSupported(type: string): Promise<boolean> {
      void type;
      return Promise.resolve(supported);
    }

    readonly complete = true;
    readonly completed = Promise.resolve();
    readonly tracks = tracks;
    readonly type: string;

    constructor(init: ImageDecoderInit) {
      this.type = init.type;
    }

    close(): void {
      closeDecoder();
    }

    decode(options?: ImageDecodeOptions): Promise<ImageDecodeResult> {
      void options;
      return decode();
    }

    reset(): void {}
  }

  class StubOffscreenCanvas {
    readonly height: number;
    readonly width: number;

    constructor(width: number, height: number) {
      this.width = width;
      this.height = height;
    }

    getContext(): OffscreenCanvasRenderingContext2D {
      return {
        drawImage: vi.fn(),
        getImageData: vi.fn(() => ({ data: pixels })),
      } as unknown as OffscreenCanvasRenderingContext2D;
    }
  }

  vi.stubGlobal("ImageDecoder", StubImageDecoder);
  vi.stubGlobal("OffscreenCanvas", StubOffscreenCanvas);
  return { closeDecoder, closeFrame, decode, pixels };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("selectImageResponse", () => {
  it.each([
    ["image/png", "image/png"],
    [" Image/JPEG ; charset=binary", "image/jpeg"],
    ["image/svg+xml", "image/svg+xml"],
    ["image/x-firefox-future-format", "image/x-firefox-future-format"],
  ] as const)("selects declared image MIME type %s", (contentType, expected) => {
    vi.spyOn(console, "debug").mockImplementation(() => undefined);

    expect(selectImageResponse(contentType)).toEqual({
      kind: "eligible",
      mimeType: expected,
    });
  });

  it.each([null, "", "image/", "application/octet-stream"])(
    "bypasses non-image Content-Type %s",
    (contentType) => {
      vi.spyOn(console, "debug").mockImplementation(() => undefined);

      expect(selectImageResponse(contentType)).toEqual({ kind: "ineligible" });
    },
  );
});

describe("decodeImage", () => {
  it("returns unpremultiplied RGBA8 sRGB pixels and releases decoder resources", async () => {
    const debugLog = vi.spyOn(console, "debug").mockImplementation(() => undefined);
    const durationMatcher: unknown = expect.any(Number);
    const browser = installBrowserStubs();

    const decoded = await decodeImage(new Uint8Array([1, 2, 3]), "image/png");

    expect(decoded).toEqual({
      width: 1,
      height: 1,
      data: browser.pixels,
      channelOrder: "RGBA",
      colorSpace: "srgb",
      alphaMode: "unpremultiplied",
    });
    expect(decoded.data).toBe(browser.pixels);
    expect(browser.decode).toHaveBeenCalledExactlyOnceWith();
    expect(browser.closeFrame).toHaveBeenCalledExactlyOnceWith();
    expect(browser.closeDecoder).toHaveBeenCalledExactlyOnceWith();
    expect(debugLog).toHaveBeenCalledExactlyOnceWith(
      "Image response decoded into the in-memory pixel boundary",
      {
        durationMilliseconds: durationMatcher,
        encodedBytes: 3,
        mimeType: "image/png",
        width: 1,
        height: 1,
      },
    );
  });

  it("rejects every animated track before decoding a frame", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const browser = installBrowserStubs({ animated: true });

    await expect(
      decodeImage(new Uint8Array([1, 2, 3]), "image/gif"),
    ).rejects.toMatchObject({
      code: ImageDecodingErrorCode.ANIMATED_IMAGE,
      message: "Animated image responses are not eligible for classification",
    });
    expect(browser.decode).not.toHaveBeenCalled();
    expect(browser.closeFrame).not.toHaveBeenCalled();
    expect(browser.closeDecoder).toHaveBeenCalledExactlyOnceWith();
  });

  it("rejects formats without an animation-aware Firefox decoder", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    installBrowserStubs({ supported: false });

    await expect(
      decodeImage(new Uint8Array([1, 2, 3]), "image/x-unknown"),
    ).rejects.toMatchObject({
      code: ImageDecodingErrorCode.UNSUPPORTED_TYPE,
      message: "Firefox cannot safely decode this image response type",
    });
  });

  it("reports malformed bytes without logging their contents", async () => {
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const durationMatcher: unknown = expect.any(Number);
    installBrowserStubs({
      decodeError: new Error("sensitive-byte-sentinel"),
    });

    await expect(
      decodeImage(new Uint8Array([1, 2, 3]), "image/png"),
    ).rejects.toMatchObject({
      code: ImageDecodingErrorCode.DECODE_FAILED,
      message: "Image response could not be decoded",
    });
    expect(errorLog).toHaveBeenCalledExactlyOnceWith(
      "Image decoding stopped: DECODE_FAILED",
      {
        durationMilliseconds: durationMatcher,
        encodedBytes: 3,
        mimeType: "image/png",
      },
    );
  });
});
