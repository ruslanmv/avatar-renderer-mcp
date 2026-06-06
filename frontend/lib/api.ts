// lib/api.ts
// ---------------------------------------------------------------------------
// Thin API client for the Avatar Renderer backend.
//
// Auth model: the browser only ever stores OUR app session token (never the
// Hugging Face access token). The backend issues it after OAuth and redirects
// back to the frontend with ?session_token=... which we capture once on load.
// ---------------------------------------------------------------------------

export const API_BASE =
  process.env.NEXT_PUBLIC_AVATAR_API_BASE || 'http://localhost:8000';

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
 * Capture `?session_token=...` from the URL after the OAuth redirect, persist
 * it, and strip it from the address bar. Call once on app mount.
 */
export function captureSessionFromUrl(): void {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  const token = url.searchParams.get('session_token');
  if (token) {
    setSessionToken(token);
    url.searchParams.delete('session_token');
    window.history.replaceState({}, '', url.toString());
  }
}

export function authHeaders(): Record<string, string> {
  const token = getSessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ----- auth endpoints ------------------------------------------------------

export function getLoginUrl(): string {
  return `${API_BASE}/auth/huggingface/login`;
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
