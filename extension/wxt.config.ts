import { defineConfig } from 'wxt';

export default defineConfig({
  manifest: {
    browser_specific_settings: {
      gecko: {
        data_collection_permissions: {
          required: ['none'],
        },
      },
    },
  },
});
