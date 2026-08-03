import { defineConfig } from 'wxt';

import baseConfig from './wxt.config';

/** Packages the Firefox extension with its installable `.xpi` filename. */
export default defineConfig({
  ...baseConfig,
  zip: {
    ...baseConfig.zip,
    artifactTemplate:
      '{{name}}-{{packageVersion}}-{{browser}}{{modeSuffix}}.xpi',
  },
});
