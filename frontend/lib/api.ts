// lib/api.ts
// ---------------------------------------------------------------------------
// Thin API client for the Avatar Renderer backend.
//
// Auth model: the browser only ever stores OUR app session token (never the
// Hugging Face access token). The backend issues it after OAuth and redirects
// back to the frontend with ?session_token=... which we capture once on load.
// ---------------------------------------------------------------------------

// Defaults to the deployed Hugging Face Space so the Vercel app works even
// without setting NEXT_PUBLIC_AVATAR_API_BASE. Override it for local dev.
export const API_BASE =
  process.env.NEXT_PUBLIC_AVATAR_API_BASE ||
  'https://ruslanmv-avatar-renderer.hf.space';

const SESSION_KEY = 'avatar_session_token';

// ----- session token storage (localStorage) -------------------------------

export function getSessionToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(SESSION_KEY);
}

export function setSessionToken(token: string): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(SESSION_KEY, token);
}

export function clearSessionToken(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(SESSION_KEY);
}

/**
 * Capture `?session_token=...` (and `?signup=`) from the URL after the OAuth
 * redirect, persist the token, and strip both from the address bar.
 * Returns whether a token was captured and whether this was a first-time signup.
 * Call once on app mount.
 */
export function captureSessionFromUrl(): { captured: boolean; signup: boolean } {
  if (typeof window === 'undefined') return { captured: false, signup: false };
  const url = new URL(window.location.href);
  const token = url.searchParams.get('session_token');
  const signup = url.searchParams.get('signup') === '1';
  if (token) {
    setSessionToken(token);
    url.searchParams.delete('session_token');
    url.searchParams.delete('signup');
    window.history.replaceState({}, '', url.toString());
    return { captured: true, signup };
  }
  return { captured: false, signup: false };
}

export function authHeaders(): Record<string, string> {
  const token = getSessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ----- auth endpoints ------------------------------------------------------

/**
 * Build the backend login URL, passing this deployment's own origin as
 * `return_to` so the backend redirects back HERE after OAuth — works across all
 * Vercel preview/production domains without backend reconfiguration.
 */
export function getLoginUrl(returnTo?: string): string {
  const origin =
    returnTo ?? (typeof window !== 'undefined' ? window.location.origin : '');
  const qs = origin ? `?return_to=${encodeURIComponent(origin)}` : '';
  return `${API_BASE}/auth/huggingface/login${qs}`;
}

export interface MeResponse {
  authenticated: boolean;
  authEnabled: boolean;
  id?: number;
  hfUserId?: string;
  username?: string | null;
  email?: string | null;
}

export async function fetchMe(): Promise<MeResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/me`, { headers: authHeaders() });
    if (res.status === 401) return { authenticated: false, authEnabled: true };
    if (!res.ok) return null;
    return (await res.json()) as MeResponse;
  } catch {
    return null;
  }
}

export async function logout(): Promise<void> {
  try {
    await fetch(`${API_BASE}/logout`, { method: 'POST', headers: authHeaders() });
  } finally {
    clearSessionToken();
  }
}

export interface RenderJobRecord {
  job_id: string;
  status: string;
  created_at: string;
  completed_at: string | null;
}

/** Fetch the signed-in user's render history (workspace-scoped). */
export async function fetchMyJobs(): Promise<RenderJobRecord[]> {
  try {
    const res = await fetch(`${API_BASE}/me/jobs`, { headers: authHeaders() });
    if (!res.ok) return [];
    const body = await res.json();
    return (body.jobs ?? []) as RenderJobRecord[];
  } catch {
    return [];
  }
}

// ----- render endpoints ----------------------------------------------------

export interface RenderJob {
  jobId: string;
  statusUrl: string;
  async: boolean;
}

export async function renderUpload(form: FormData): Promise<RenderJob> {
  const res = await fetch(`${API_BASE}/render-upload`, {
    method: 'POST',
    headers: authHeaders(), // do NOT set Content-Type; browser sets multipart boundary
    body: form,
  });
  if (res.status === 401) {
    throw new Error('Please sign in with Hugging Face before rendering.');
  }
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  return (await res.json()) as RenderJob;
}

/** One status poll. Returns an MP4 blob when ready, else a state string. */
export async function pollStatusOnce(
  jobId: string,
): Promise<{ done: true; blob: Blob } | { done: false; state: string }> {
  const res = await fetch(`${API_BASE}/status/${jobId}`, { headers: authHeaders() });
  const ct = res.headers.get('content-type') || '';
  if (res.ok && ct.includes('video/mp4')) {
    return { done: true, blob: await res.blob() };
  }
  let state = res.ok ? 'processing' : `error_${res.status}`;
  try {
    const body = await res.json();
    if (body?.state) state = body.state;
  } catch {
    /* not JSON */
  }
  return { done: false, state };
}
