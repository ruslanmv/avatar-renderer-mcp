'use client';

// components/HuggingFaceLogin.tsx
// ---------------------------------------------------------------------------
// Sign-in widget. Shows a "Sign in with Hugging Face" button when logged out,
// and the username + "Sign out" when logged in.
//
// When the backend reports AUTH is disabled (authEnabled === false), the widget
// renders nothing — so the same frontend works against an open dev backend and
// a protected production backend with no code changes.
// ---------------------------------------------------------------------------

import { useEffect, useState } from 'react';
import { LogIn, LogOut } from 'lucide-react';
import {
  captureSessionFromUrl,
  fetchMe,
  getLoginUrl,
  logout,
  type MeResponse,
} from '../lib/api';

export default function HuggingFaceLogin() {
  const [me, setMe] = useState<MeResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    captureSessionFromUrl();
    fetchMe()
      .then(setMe)
      .finally(() => setLoading(false));
  }, []);

  // Hide entirely when auth is turned off on the backend, or while loading.
  if (loading || !me || me.authEnabled === false) return null;

  const handleLogout = async () => {
    await logout();
    setMe({ authenticated: false, authEnabled: true });
  };

  if (me.authenticated) {
    return (
      <div className="flex items-center gap-3">
        <span className="text-sm text-cyan-200">
          {me.username ? `@${me.username}` : 'Signed in'}
        </span>
        <button
          onClick={handleLogout}
          className="flex items-center gap-1.5 rounded-lg border border-cyan-500/30 px-3 py-1.5 text-sm text-cyan-100 transition hover:bg-cyan-500/10"
        >
          <LogOut size={15} />
          Sign out
        </button>
      </div>
    );
  }

  return (
    <a
      href={getLoginUrl()}
      className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-yellow-400 to-amber-500 px-4 py-2 text-sm font-semibold text-black transition hover:opacity-90"
    >
      <LogIn size={16} />
      Sign in with Hugging Face
    </a>
  );
}
