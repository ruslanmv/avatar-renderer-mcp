'use client';

// components/HuggingFaceLogin.tsx
// ---------------------------------------------------------------------------
// Sign-in widget driven by the shared auth context (lib/auth.tsx).
//
//  • logged out → "Sign in with Hugging Face"
//  • logged in  → "@username" + "Sign out" (and a one-time welcome on signup)
//  • auth disabled on the backend → renders nothing, so the same build works
//    against an open dev backend and a protected production backend.
// ---------------------------------------------------------------------------

import { LogIn, LogOut } from 'lucide-react';
import { useAuth } from '../lib/auth';

export default function HuggingFaceLogin() {
  const { loading, authEnabled, isAuthenticated, user, justSignedUp, login, logout } = useAuth();

  // Hide entirely while loading or when auth is turned off on the backend.
  if (loading || !authEnabled) return null;

  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-3">
        <span className="text-sm text-cyan-200">
          {justSignedUp ? 'Welcome, ' : ''}
          {user?.username ? `@${user.username}` : 'Signed in'}
        </span>
        <button
          onClick={logout}
          className="flex items-center gap-1.5 rounded-lg border border-cyan-500/30 px-3 py-1.5 text-sm text-cyan-100 transition hover:bg-cyan-500/10"
        >
          <LogOut size={15} />
          Sign out
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={login}
      className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-yellow-400 to-amber-500 px-4 py-2 text-sm font-semibold text-black transition hover:opacity-90"
    >
      <LogIn size={16} />
      Sign in with Hugging Face
    </button>
  );
}
