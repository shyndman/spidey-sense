import { defineConfig } from 'vitest/config';

import { byteImportPlugin } from './vite/byte-import-plugin.ts';

export default defineConfig({
  plugins: [byteImportPlugin()],
});
