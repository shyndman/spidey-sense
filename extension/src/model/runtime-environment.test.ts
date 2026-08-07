import { env } from 'onnxruntime-web/wasm';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  configureOnnxRuntime,
  initializeOnnxRuntime,
} from './runtime-environment';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('configureOnnxRuntime', () => {
  it('selects the packaged single-threaded WASM runtime without proxying', () => {
    configureOnnxRuntime();

    expect(env.wasm.numThreads).toBe(1);
    expect(env.wasm.proxy).toBe(false);

    const paths = env.wasm.wasmPaths;
    if (
      typeof paths !== 'object' ||
      paths.wasm === undefined ||
      paths.mjs === undefined
    ) {
      throw new Error('Expected explicit packaged ONNX Runtime URLs');
    }
    expect(paths.wasm).toBeInstanceOf(URL);
    expect(paths.mjs).toBeInstanceOf(URL);
    expect(paths.wasm.toString()).not.toMatch(/^https?:/);
    expect(paths.mjs.toString()).not.toMatch(/^https?:/);
  });
});

describe('initializeOnnxRuntime', () => {
  it('configures the runtime before invoking session initialization', async () => {
    env.wasm.wasmPaths = undefined;

    await expect(
      initializeOnnxRuntime(() => {
        expect(env.wasm.wasmPaths).toBeDefined();
        return Promise.resolve('initialized' as const);
      }),
    ).resolves.toBe('initialized');
  });

  it('traces a sanitized failure and preserves its cause', async () => {
    const cause = new Error('private asset location');
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => undefined);

    const result = initializeOnnxRuntime(() => Promise.reject(cause));

    await expect(result).rejects.toMatchObject({
      name: 'RuntimeInitializationError',
      message: 'ONNX Runtime failed to initialize',
      cause,
    });
    expect(errorLog).toHaveBeenCalledExactlyOnceWith(
      'ONNX Runtime initialization failed',
    );
  });
});
