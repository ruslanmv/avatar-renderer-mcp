// lib/hfToken.ts
// ---------------------------------------------------------------------------
// Stores the user's Hugging Face access token (browser-side) so inference runs
// on THEIR ZeroGPU quota. Two ways to obtain it:
//   1) Paste a token (huggingface.co/settings/tokens) — works immediately.
//   2) HF OAuth (productionization) — set NEXT_PUBLIC_OAUTH_CLIENT_ID and use
//      @huggingface/hub's oauth helpers; the resulting accessToken is stored here.
// The token is kept only in localStorage and sent directly to the Space.
// ---------------------------------------------------------------------------

const KEY = 'hf_token';

export function getHfToken(): string | null {
  if (typeof window === 'undefined') return null;
  const t = window.localStorage.getItem(KEY);
  return t && t.trim() ? t.trim() : null;
}

export function setHfToken(token: string): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(KEY, token.trim());
}

export function clearHfToken(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(KEY);
}

export function hasHfToken(): boolean {
  return Boolean(getHfToken());
}
