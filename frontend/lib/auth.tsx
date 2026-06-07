'use client';

// lib/auth.tsx
// ---------------------------------------------------------------------------
// App-wide auth state for the per-user multitenant frontend.
//
// Wrap the app in <AuthProvider> (see app/layout.tsx) and read state anywhere
// with useAuth(). On mount it captures the session token from the OAuth
// redirect, then loads the current user from the backend's /me endpoint.
// ---------------------------------------------------------------------------

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import {
  captureSessionFromUrl,
  fetchMe,
  getLoginUrl,
  logout as apiLogout,
  type MeResponse,
} from './api';

interface AuthState {
  user: MeResponse | null;
  loading: boolean;
  /** True when the backend has auth turned on. */
  authEnabled: boolean;
  /** True if the user is signed in. */
  isAuthenticated: boolean;
  /** True for the redirect immediately following a first-time signup. */
  justSignedUp: boolean;
  login: () => void;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<MeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [justSignedUp, setJustSignedUp] = useState(false);

  const refresh = useCallback(async () => {
    const me = await fetchMe();
    setUser(me);
  }, []);

  useEffect(() => {
    const { signup } = captureSessionFromUrl();
    setJustSignedUp(signup);
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  const login = useCallback(() => {
    // Send this deployment's origin so the backend returns here after OAuth.
    window.location.href = getLoginUrl();
  }, []);

  const logout = useCallback(async () => {
    await apiLogout();
    setUser({ authenticated: false, authEnabled: true });
    setJustSignedUp(false);
  }, []);

  const value = useMemo<AuthState>(
    () => ({
      user,
      loading,
      authEnabled: user?.authEnabled ?? false,
      isAuthenticated: Boolean(user?.authenticated),
      justSignedUp,
      login,
      logout,
      refresh,
    }),
    [user, loading, justSignedUp, login, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within <AuthProvider>');
  }
  return ctx;
}
