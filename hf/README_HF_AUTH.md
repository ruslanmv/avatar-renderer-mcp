# Hugging Face Login + SaaS Demo Deployment

This guide turns the existing backend into a small SaaS-style demo:

```
User browser → Vercel frontend → FastAPI backend on a Hugging Face GPU Space → render_pipeline() → MP4
```

The frontend (Vercel) hosts **no GPU**. All inference runs in the Hugging Face
Space. Authentication is **Hugging Face OAuth**, and users/sessions/jobs are
stored in **SQLite** inside the backend.

> Auth is **opt-in**. With `AUTH_ENABLED` unset/false the backend behaves exactly
> as before (no login required) — useful for local dev and CI. Set
> `AUTH_ENABLED=true` on the Space to require sign-in.

---

## 1. What was added

| Area | File(s) | Purpose |
|---|---|---|
| OAuth client | `app/auth_hf.py` | Build authorize URL, exchange code, fetch userinfo |
| Data store | `app/users_db.py` | SQLite `users` / `sessions` / `render_jobs` |
| Token helpers | `app/security.py` | Session tokens + signed OAuth CSRF state |
| Endpoints | `app/api.py` | `/auth/huggingface/login`, `/auth/huggingface/callback`, `/me`, `/logout`; auth-gated `/render-upload`, `/render`, `/status/{id}` |
| Frontend | `frontend/lib/api.ts`, `frontend/components/HuggingFaceLogin.tsx` | Login button, session capture, authenticated fetches |
| Startup | `hf/start.sh` | Persistent SQLite path on `/data` |

No new heavy dependencies: the backend uses the stdlib `sqlite3` plus
`requests` (already installed in `hf/Dockerfile`).

---

## 2. Create a Hugging Face OAuth app

1. Go to **https://huggingface.co/settings/applications** → *New application*
   (a "Connected App"). For Spaces OAuth see
   https://huggingface.co/docs/hub/spaces-oauth.
2. Set the **Redirect URI** to your Space callback:
   ```
   https://<your-space-subdomain>.hf.space/auth/huggingface/callback
   ```
3. Scopes: `openid profile email`.
4. Copy the **Client ID** and **Client Secret**.

---

## 3. Configure the Space (as **Secrets**, not variables)

In your Space → *Settings* → *Variables and secrets*:

| Name | Example | Secret? |
|---|---|---|
| `AUTH_ENABLED` | `true` | no |
| `HF_CLIENT_ID` | `xxxxxxxx` | yes |
| `HF_CLIENT_SECRET` | `xxxxxxxx` | **yes** |
| `HF_REDIRECT_URI` | `https://<space>.hf.space/auth/huggingface/callback` | no |
| `FRONTEND_URL` | `https://<your-app>.vercel.app` | no |
| `SESSION_SECRET` | a long random string | **yes** |
| `CORS_ALLOW_ORIGINS` | `https://<your-app>.vercel.app` | no |
| `DATABASE_URL` | `/data/avatar_app.sqlite3` | no |

> Enable **persistent storage** on the Space so `/data` (SQLite + model cache)
> survives restarts. Without it, the DB falls back to `/tmp` (ephemeral).

Never expose `HF_CLIENT_SECRET` or any user's HF token to the frontend.

---

## 4. Configure the Vercel frontend

Set one environment variable in Vercel:

```
NEXT_PUBLIC_AVATAR_API_BASE = https://<your-space-subdomain>.hf.space
```

Then deploy. The login button appears automatically when the backend reports
`authEnabled: true` (and hides itself when auth is disabled).

---

## 5. The login flow

```
Frontend  "Sign in with Hugging Face"
   │  → GET  {API}/auth/huggingface/login
   ▼
Backend redirects to huggingface.co/oauth/authorize  (sets a signed state cookie)
   │  user authorizes
   ▼
HF → GET {API}/auth/huggingface/callback?code=...&state=...
   │  backend verifies state, exchanges code, fetches userinfo,
   │  upserts user, creates a session token
   ▼
Backend → 302 redirect to  {FRONTEND_URL}?session_token=<opaque>
   │  frontend stores token in localStorage, strips it from the URL
   ▼
Authenticated requests send  Authorization: Bearer <session_token>
```

Endpoints:

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/auth/huggingface/login` | — | Redirects to HF |
| GET | `/auth/huggingface/callback` | — | Issues session, redirects to frontend |
| GET | `/me` | optional | `{authenticated, authEnabled, username, …}` |
| POST | `/logout` | bearer | Invalidates the session |
| POST | `/render-upload` | required* | Browser render path |
| POST | `/render` | required* | Trusted server-paths path |
| GET | `/status/{id}` | required* | Owner-only when auth on |

\* required only when `AUTH_ENABLED=true`.

---

## 6. GPU strategy

**Option A — recommended first:** run this FastAPI backend in a **HF Docker
Space with a GPU**. Your existing `hf/Dockerfile` + `hf/start.sh` already do
this; the Space owns the GPU cost. This is the fastest path to a working demo.

**Option B — later:** a separate **ZeroGPU Gradio** Space (`zerogpu/`) that wraps
`render_pipeline` with `@spaces.GPU`. The backend can call it with the user's HF
token so the user's own ZeroGPU quota applies. Riskier (heavy deps, cold starts),
so treat it as an optimization, not the MVP.

---

## 7. Security checklist (demo-grade)

- ✅ Login required for `/render-upload`, `/render`, `/status` (when `AUTH_ENABLED`).
- ✅ Users can only read their own jobs (ownership enforced in `/status`).
- ✅ Only the **app session token** is stored in the browser — never the HF token.
- ✅ `HF_CLIENT_SECRET` and HF tokens live only on the backend.
- ✅ OAuth CSRF state is signed with `SESSION_SECRET` and checked on callback.
- ⚠️ For production: encrypt `hf_access_token` at rest (or stop storing it),
  add upload size/type limits and rate limiting, and rotate `SESSION_SECRET`.
- ⚠️ For public SaaS, prefer `/render-upload` and keep `/render` (server paths)
  for trusted callers only.

---

## 8. Local development

```bash
# Backend (auth off — default)
make run                       # http://localhost:8000

# Backend (auth on, local OAuth app pointing at localhost)
AUTH_ENABLED=true \
HF_CLIENT_ID=... HF_CLIENT_SECRET=... \
HF_REDIRECT_URI=http://localhost:8000/auth/huggingface/callback \
FRONTEND_URL=http://localhost:3000 SESSION_SECRET=dev-secret \
python -m uvicorn app.api:app --port 8000

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```
