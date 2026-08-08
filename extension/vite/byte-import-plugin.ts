import { readFileSync } from 'node:fs';

import type { Plugin } from 'vite';

const BYTES_QUERY = '?bytes';

export function byteImportPlugin(): Plugin {
  return {
    name: 'byte-import-bytes',
    enforce: 'pre',
    load(id) {
      if (!id.endsWith(BYTES_QUERY)) return;

      const sourcePath = id.slice(0, -BYTES_QUERY.length);
      this.addWatchFile(sourcePath);

      // Embed exact file bytes during bundling so intercept-proof responses stay deterministic.
      const bytes = Array.from(readFileSync(sourcePath));
      return `export default new Uint8Array([${bytes.join(',')}]);`;
    },
  };
}
