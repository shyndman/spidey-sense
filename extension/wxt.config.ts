import { readdirSync } from 'node:fs';
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

const MODEL_METADATA_SUFFIX = '.metadata.json';
const publicModelsDirectory = fileURLToPath(
  new URL('./public/models/', import.meta.url),
);
const [modelMetadataFilename, additionalModelMetadataFilename] = readdirSync(
  publicModelsDirectory,
).filter((filename) => filename.endsWith(MODEL_METADATA_SUFFIX));
if (
  modelMetadataFilename === undefined ||
  additionalModelMetadataFilename !== undefined
) {
  throw new Error('Expected exactly one generated model metadata artifact');
}
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
      'import.meta.env.WXT_MODEL_METADATA_PATH': JSON.stringify(
        `models/${modelMetadataFilename}`,
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
    permissions: ['webRequest', 'webRequestBlocking'],
    host_permissions: [
      // Reddit
      'https://external-preview.redd.it/*',
      'https://preview.redd.it/*',
      'https://i.redd.it/*',
      'https://www.reddit.com/*',

      // YouTube
      'https://www.youtube.com/*',
      'https://yt3.ggpht.com/*',
    ],
    content_security_policy: {
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    },
    browser_specific_settings: {
      gecko: {
        id: 'spidey-sense@shyndman.ca',
        data_collection_permissions: {
          required: ['none'],
        },
      },
    },
  },
});
