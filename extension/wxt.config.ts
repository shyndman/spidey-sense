import { copyFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'wxt';

const wasmSource = fileURLToPath(
  import.meta.resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'),
);
const wasmModuleSource = fileURLToPath(
  import.meta.resolve('onnxruntime-web/ort-wasm-simd-threaded.mjs'),
);
const runtimeAssets = [
  { filename: basename(wasmSource), source: wasmSource },
  { filename: basename(wasmModuleSource), source: wasmModuleSource },
] as const;
const [wasmAsset, wasmModuleAsset] = runtimeAssets;

export default defineConfig({
  vite: () => ({
    define: {
      'import.meta.env.WXT_ONNX_WASM_FILENAME': JSON.stringify(
        wasmAsset.filename,
      ),
      'import.meta.env.WXT_ONNX_WASM_MODULE_FILENAME': JSON.stringify(
        wasmModuleAsset.filename,
      ),
    },
    resolve: {
      conditions: ['onnxruntime-web-use-extern-wasm'],
    },
  }),
  hooks: {
    'prepare:publicPaths': (_wxt, paths) => {
      for (const asset of runtimeAssets) paths.push(asset.filename);
    },
    'build:done': async (wxt, output) => {
      await Promise.all(
        runtimeAssets.map(async (asset) => {
          await copyFile(
            asset.source,
            resolve(wxt.config.outDir, asset.filename),
          );
          output.publicAssets.push({ type: 'asset', fileName: asset.filename });
        }),
      );
    },
  },
  manifest: {
    content_security_policy: {
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    },
    browser_specific_settings: {
      gecko: {
        data_collection_permissions: {
          required: ['none'],
        },
      },
    },
  },
});
