# Sign-in / Sign-up & Per-User Multitenancy

This document is the complete design for authentication and tenant isolation in
the Avatar Renderer SaaS demo, and the exact steps to make login work safely on
your Vercel deployment.

**Decisions implemented**

| Decision | Choice |
|---|---|
| Sign-in / sign-up | **Hugging Face OAuth only** (no passwords) |
| Tenancy model | **Per-user workspace** (each user is an isolated tenant) |
| Backend host | **Hugging Face Space** (FastAPI + SQLite) |
| Frontend host | **Vercel** (Next.js) |
| Session transport | **Bearer token** in `Authorization` header (no cross-site cookies) |

---

## 1. Sign-up vs sign-in

With OAuth there is one flow; the backend distinguishes the two by whether the
account already exists:

```
"Sign in with Hugging Face"
    → /auth/huggingface/login            (backend)
    → huggingface.co/oauth/authorize     (user authorizes)
    → /auth/huggingface/callback         (backend: exchange code, read profile)
         • account is NEW  → SIGN-UP  (row created, signup=1)
         • account EXISTS  → SIGN-IN  (row updated, signup=0)
    → redirect back to the frontend with ?session_token=…&signup=…
```

`upsert_user()` creates the account on first login and updates it on return;
`get_user_by_hf_id()` is checked first so the callback can flag `signup=1` for a
one-time welcome in the UI.

---

## 2. Per-user multitenancy

Each authenticated user **is** a tenant ("workspace"). Every tenant-scoped
resource carries `user_id`, and every read/write is filtered by the
authenticated user. There is no endpoint that returns another tenant's data.

```
users(id, hf_user_id, hf_username, hf_email, hf_access_token, created_at, updated_at)
sessions(id, user_id→users.id, session_token, expires_at, created_at)
render_jobs(id, user_id→users.id, job_id, status, created_at, completed_at)
```

**Isolation invariants (enforced in `app/api.py`)**

- `/render-upload`, `/render` — require a valid session; the created job is
  recorded with the caller's `user_id`.
- `/status/{job_id}` — 403 if the job's owner ≠ the caller.
- `/me/jobs` — returns only `render_jobs` where `user_id` = caller.
- `/me` — returns the caller's profile + `workspaceId` (= their user id).

`workspaceId` mirrors `user_id` today. When organizations are added later, only
the resolution of "active tenant" changes — the per-row `user_id`/`tenant_id`
filter and the access checks stay the same. (See §7.)

---

## 3. Session security

- Session token: 256-bit, URL-safe, opaque (`secrets.token_urlsafe`), stored in
  `sessions` with an expiry (`SESSION_TTL_HOURS`, default 7 days). Expired
  tokens are rejected and deleted on access.
- Transport: `Authorization: Bearer <token>`. The browser stores **only** this
  app token (in `localStorage`) — never the Hugging Face access token, which
  stays server-side in `users.hf_access_token`.
- OAuth CSRF: a per-login `state` is HMAC-signed with `SESSION_SECRET` and also
  set as a first-party cookie on the backend domain; the callback requires both
  to match.
- Open-redirect protection: the post-login `return_to` is validated against the
  CORS allowlist/regex before the backend will redirect to it (see §5).

---

## 4. Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/auth/huggingface/login?return_to=<origin>` | — | Redirect to HF authorize |
| GET | `/auth/huggingface/callback` | — | Issue session, redirect to frontend |
| GET | `/me` | optional | `{authenticated, authEnabled, workspaceId, username, email}` |
| GET | `/me/jobs` | required* | The user's render history |
| POST | `/logout` | bearer | Invalidate the session |
| POST | `/render-upload` | required* | Browser render path |
| POST | `/render` | required* | Trusted server-paths path |
| GET | `/status/{id}` | required* | Owner-only |

\* required only when `AUTH_ENABLED=true`. With auth off, everything behaves as
the pre-auth backend (keeps local dev and CI unchanged).

---

## 5. Making login work on your Vercel domain (the important part)

Your deployment URL is a **per-commit Vercel preview**:

```
https://avatar-renderer-git-257f9d-ruslan-magana-vsevolodovnas-projects.vercel.app/
```

The `…git-257f9d…` part changes on every push, so we do **not** hardcode it.
Instead:

1. The frontend sends its **own** `window.location.origin` as `return_to` when
   it calls `/auth/huggingface/login`.
2. The backend validates that origin against `CORS_ALLOW_ORIGINS` (exact list)
   **or** `CORS_ALLOW_ORIGIN_REGEX` (default `https://.*\.vercel\.app`, which
   matches the URL above and every other `*.vercel.app` preview/prod domain).
3. After OAuth it redirects back to that exact origin with the session token.

So **any** of your Vercel domains can log in with zero backend changes, while
arbitrary external origins are refused (no open redirect).

### One-time setup

**A. Hugging Face OAuth app** — https://huggingface.co/settings/applications

- Redirect URI (points at the **backend**, which has a stable domain):
  ```
  https://<your-space-subdomain>.hf.space/auth/huggingface/callback
  ```
- Scopes: `openid profile email`. Copy the Client ID and Client Secret.

**B. Hugging Face Space — Settings → Variables and secrets**

| Name | Value | Secret? |
|---|---|---|
| `AUTH_ENABLED` | `true` | no |
| `HF_CLIENT_ID` | … | yes |
| `HF_CLIENT_SECRET` | … | **yes** |
| `HF_REDIRECT_URI` | `https://<space>.hf.space/auth/huggingface/callback` | no |
| `SESSION_SECRET` | long random string | **yes** |
| `FRONTEND_URL` | your **stable** prod domain (fallback redirect) | no |
| `CORS_ALLOW_ORIGIN_REGEX` | `https://.*\.vercel\.app` (default; keep) | no |
| `CORS_ALLOW_ORIGINS` | optionally pin exact prod domain(s) | no |
| `DATABASE_URL` | `/data/avatar_app.sqlite3` (enable persistent storage) | no |

> To restrict to only your project's domains, set
> `CORS_ALLOW_ORIGIN_REGEX=https://avatar-renderer-.*\.vercel\.app` instead of
> the broad default.

**C. Vercel — Project → Settings → Environment Variables**

```
NEXT_PUBLIC_AVATAR_API_BASE = https://<your-space-subdomain>.hf.space
```

Redeploy the frontend. The "Sign in with Hugging Face" button appears
automatically (it hides itself when `AUTH_ENABLED` is false).

---

## 6. End-to-end flow

```
Vercel frontend (this preview URL)
  │  click "Sign in with Hugging Face"
  │  GET {API}/auth/huggingface/login?return_to=https://…vercel.app
  ▼
HF Space backend → 302 huggingface.co/oauth/authorize  (signed state + return_to cookies)
  │  user authorizes
  ▼
HF → GET {API}/auth/huggingface/callback?code&state
  │  verify state, exchange code, read profile, upsert user, create session
  ▼
backend → 302 https://…vercel.app/?session_token=…&signup=…
  │  AuthProvider captures token → localStorage, strips the query, loads /me
  ▼
authenticated: render/upload/status calls send Authorization: Bearer <token>
```

---

## 7. Upgrading to organizations later (not built yet)

The per-user model is forward-compatible. To add orgs:

1. Add `organizations` and `memberships(user_id, org_id, role)` tables.
2. Add `org_id` to `render_jobs`; scope queries by the **active** org instead of
   (or in addition to) `user_id`.
3. `/me` returns the membership list + active org; add an org switcher in the UI.
4. Replace the ownership check in `/status` with an org-membership check.

No change to the OAuth flow, session model, or the Vercel wiring is required.

---

## 8. Security checklist (demo-grade)

- ✅ Login required for render/status when `AUTH_ENABLED`.
- ✅ Strict per-user data isolation (jobs, history).
- ✅ Only the app session token reaches the browser; HF token stays server-side.
- ✅ HMAC-signed OAuth state + validated `return_to` (no CSRF, no open redirect).
- ✅ Session expiry + logout invalidation.
- ⚠️ Production hardening: encrypt `hf_access_token` at rest (or don't store it),
  add upload size/type limits and per-user rate limiting, rotate `SESSION_SECRET`,
  and move SQLite → Postgres/Turso for durability and concurrency.
