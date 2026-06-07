// lib/gradioClient.ts
// ---------------------------------------------------------------------------
// Runs inference against the Hugging Face ZeroGPU Gradio Space's /predict API.
// The browser uploads the image + audio directly to the Space and gets back a
// video URL — Vercel hosts no GPU; all inference happens on the Space.
// ---------------------------------------------------------------------------

import { Client } from '@gradio/client';

// "owner/space" id, or a full https URL to the Space.
export const HF_SPACE =
  process.env.NEXT_PUBLIC_HF_SPACE || 'ruslanmv/avatar-renderer';

/**
 * Generate a talking-avatar video on the GPU backend.
 * @returns a playable video URL hosted on the Space.
 */
export async function generateAvatar(
  image: File | Blob,
  audio: File | Blob,
  qualityMode = 'auto',
): Promise<string> {
  const client = await Client.connect(HF_SPACE);

  // Positional args match space_app.generate(image_path, audio_path, quality_mode)
  const result: any = await client.predict('/predict', [image, audio, qualityMode]);

  // Gradio 5's Video output is { video: <FileData|str>, subtitles }. Normalize
  // across shapes: {url}, {path}, "url", { video: {url|path} | "url" }.
  const out = Array.isArray(result?.data) ? result.data[0] : result?.data;
  const v = out?.video ?? out;
  const url =
    (typeof v === 'string' ? v : null) ||
    v?.url ||
    v?.path ||
    null;

  if (!url || !/^https?:\/\//.test(url)) {
    throw new Error('The GPU backend did not return a playable video URL.');
  }
  return url;
}
