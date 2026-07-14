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

export interface GenerateOptions {
  image: File | Blob;
  audio?: File | Blob | null;
  /** If set, synthesized to speech by the backend (edge-tts) and used as audio. */
  text?: string;
  voice?: string;
  speed?: number; // -50..50 (%)
  pitch?: number; // -20..20 (Hz)
  quality?: string;
  enhancements?: string[];
  /** Rendering method: wav2lip_gfpgan | fullface | wav2lip | simple | auto */
  method?: string;
  /** HF token → run on the user's own ZeroGPU quota. */
  hfToken?: string;
}

/**
 * Generate a talking-avatar video on the GPU backend.
 * @returns a playable video URL hosted on the Space.
 */
export async function generateAvatar(opts: GenerateOptions): Promise<string> {
  const client = await Client.connect(
    HF_SPACE,
    opts.hfToken ? { hf_token: opts.hfToken as `hf_${string}` } : undefined,
  );

  // Positional args match space_app.generate(
  //   image, audio, text, voice, speed, pitch, quality, addons)
  const result: any = await client.predict('/predict', [
    opts.image,
    opts.audio ?? null,
    opts.text ?? '',
    opts.voice ?? 'en-US-AriaNeural',
    opts.speed ?? 0,
    opts.pitch ?? 0,
    opts.quality ?? 'auto',
    opts.enhancements ?? [],
    opts.method ?? 'wav2lip_fast',
  ]);

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
