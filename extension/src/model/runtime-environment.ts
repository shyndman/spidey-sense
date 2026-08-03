import { env } from 'onnxruntime-web/wasm';

const SINGLE_RUNTIME_THREAD = 1;

const runtimeBaseUrl = globalThis.location?.href ?? import.meta.url;

/** A sanitized failure raised when the local ONNX Runtime cannot initialize. */
export class RuntimeInitializationError extends Error {
  constructor(cause: unknown) {
    super('ONNX Runtime failed to initialize', { cause });
    this.name = 'RuntimeInitializationError';
  }
}

/**
 * Configures inference to use the extension-packaged WebAssembly backend only.
 * A single thread avoids worker and SharedArrayBuffer requirements in Firefox;
 * the explicit build-generated URL prevents ONNX Runtime from falling back to
 * a CDN or resolving assets relative to a caller's script location.
 */
export function configureOnnxRuntime(): void {
  env.wasm.numThreads = SINGLE_RUNTIME_THREAD;
  env.wasm.proxy = false;
  env.wasm.wasmPaths = {
    wasm: new URL(
      import.meta.env.WXT_ONNX_WASM_FILENAME,
      runtimeBaseUrl,
    ),
    mjs: new URL(
      import.meta.env.WXT_ONNX_WASM_MODULE_FILENAME,
      runtimeBaseUrl,
    ),
  };
}

/**
 * Runs session initialization after local runtime configuration and traces only
 * a stable failure message. Callers retain the original cause for diagnostics,
 * but asset URLs, model bytes, tensors, and image data never enter the log.
 */
export async function initializeOnnxRuntime<Result>(
  initializer: () => Promise<Result>,
): Promise<Result> {
  try {
    configureOnnxRuntime();
    return await initializer();
  } catch (cause: unknown) {
    console.error('ONNX Runtime initialization failed');
    throw new RuntimeInitializationError(cause);
  }
}
