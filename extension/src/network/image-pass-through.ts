import { browser } from 'wxt/browser';

interface ImagePassThroughStreamFilter {
  readonly error: string;
  ondata: ((event: { readonly data: ArrayBuffer }) => void) | null;
  onerror: (() => void) | null;
  onstop: (() => void) | null;
  write(data: Uint8Array<ArrayBuffer>): void;
  close(): void;
}

export interface ImagePassThroughWebRequest {
  readonly onBeforeRequest: {
    addListener(
      listener: (details: {
        readonly requestId: string;
        readonly url: string;
      }) => void,
      filter: {
        readonly urls: readonly string[];
        readonly types: readonly ['image'];
      },
      extraInfoSpec: readonly ['blocking'],
    ): void;
  };
  filterResponseData(requestId: string): ImagePassThroughStreamFilter;
}

const IMAGE_HOST_PATTERNS = [
  'https://yt3.ggpht.com/*',
  'https://preview.redd.it/*',
] as const;
const IMAGE_PASS_THROUGH_EXTRA_INFO_SPEC = ['blocking'] as const;
const IMAGE_RESOURCE_TYPES = ['image'] as const;
const IMAGE_PASS_THROUGH_ERROR = 'Image response pass-through failed';
const IMAGE_PASS_THROUGH_LOG_MESSAGE = 'Intercepting image URL';
const SERVICE_WORKER_FALLBACK_REDIRECTION_ERROR =
  'ServiceWorker fallback redirection';

export function registerImagePassThrough(
  //! HACK: WXT 0.21.3 types are based on Chrome declarations and omit Firefox-only
  //! `filterResponseData`, while the runtime API is permission-gated and verified
  //! in packaged Firefox.
  webRequest: ImagePassThroughWebRequest =
    browser.webRequest as unknown as ImagePassThroughWebRequest,
): void {
  webRequest.onBeforeRequest.addListener(
    (details) => {
      // Let users see which intercepted image URLs are being passed through.
      console.info(IMAGE_PASS_THROUGH_LOG_MESSAGE, details.url);
      let filter: ImagePassThroughStreamFilter;
      try {
        filter = webRequest.filterResponseData(details.requestId);
      } catch {
        console.error(IMAGE_PASS_THROUGH_ERROR);
        return;
      }

      let chunks: ArrayBuffer[] = [];
      let totalBytes = 0;
      let terminated = false;

      filter.ondata = ({ data }) => {
        if (terminated) return;
        chunks.push(data);
        totalBytes += data.byteLength;
      };
      filter.onerror = () => {
        if (terminated) return;
        terminated = true;
        chunks = [];
        totalBytes = 0;
        if (filter.error !== SERVICE_WORKER_FALLBACK_REDIRECTION_ERROR) {
          console.error(IMAGE_PASS_THROUGH_ERROR, filter.error);
        }
      };
      filter.onstop = () => {
        if (terminated) return;
        terminated = true;

        try {
          const body = new Uint8Array(totalBytes);
          let offset = 0;
          for (const chunk of chunks) {
            body.set(new Uint8Array(chunk), offset);
            offset += chunk.byteLength;
          }
          filter.write(body);
          filter.close();
        } catch {
          console.error(IMAGE_PASS_THROUGH_ERROR);
        } finally {
          chunks = [];
          totalBytes = 0;
        }
      };
    },
    {
      urls: IMAGE_HOST_PATTERNS,
      types: IMAGE_RESOURCE_TYPES,
    },
    IMAGE_PASS_THROUGH_EXTRA_INFO_SPEC,
  );
}
