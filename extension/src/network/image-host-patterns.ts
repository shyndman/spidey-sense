/**
 * Remote image origins whose image and responsive-image responses are replaced.
 * Sharing this list with the manifest keeps every intercepted origin permissioned.
 */
export const IMAGE_HOST_PATTERNS = [
  'https://yt3.ggpht.com/*',
  'https://i.ytimg.com/*',
  'https://external-preview.redd.it/*',
  'https://preview.redd.it/*',
  'https://i.redd.it/*',
] as const;
