import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import {
  type InterceptProofImageMimeType,
  INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE,
} from './image-bytes';

const IMAGE_FIXTURES = [
  { mimeType: 'image/png', filename: 'magenta.png' },
  { mimeType: 'image/jpeg', filename: 'magenta.jpg' },
  { mimeType: 'image/gif', filename: 'magenta.gif' },
  { mimeType: 'image/webp', filename: 'magenta.webp' },
  { mimeType: 'image/svg+xml', filename: 'magenta.svg' },
] as const satisfies ReadonlyArray<{
  mimeType: InterceptProofImageMimeType;
  filename: string;
}>;

describe('INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE', () => {
  it.each(IMAGE_FIXTURES)('matches the exact $mimeType asset bytes', async ({
    mimeType,
    filename,
  }) => {
    const expectedBytes = new Uint8Array(
      await readFile(new URL(`./images/${filename}`, import.meta.url)),
    );

    expect(INTERCEPT_PROOF_IMAGE_BYTES_BY_MIME_TYPE[mimeType]).toEqual(
      expectedBytes,
    );
  });
});
