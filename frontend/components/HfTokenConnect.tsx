'use client';

// components/HfTokenConnect.tsx
// ---------------------------------------------------------------------------
// Lets a user connect their Hugging Face token so ZeroGPU inference runs on
// THEIR quota (passed to Client.connect). Token is stored only in localStorage.
// ---------------------------------------------------------------------------

import { useEffect, useState } from 'react';
import { KeyRound, Check, LogOut } from 'lucide-react';
import { getHfToken, setHfToken, clearHfToken } from '../lib/hfToken';

export default function HfTokenConnect() {
  const [connected, setConnected] = useState(false);
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState('');

  useEffect(() => {
    setConnected(Boolean(getHfToken()));
  }, []);

  const connect = () => {
    const t = value.trim();
    if (!t.startsWith('hf_')) {
      alert('Please paste a valid Hugging Face token (starts with "hf_").');
      return;
    }
    setHfToken(t);
    setConnected(true);
    setOpen(false);
    setValue('');
  };

  const disconnect = () => {
    clearHfToken();
    setConnected(false);
  };

  if (connected) {
    return (
      <div className="flex items-center gap-2">
        <span className="flex items-center gap-1 text-sm text-emerald-300">
          <Check size={15} /> HF token connected
        </span>
        <button
          onClick={disconnect}
          className="flex items-center gap-1.5 rounded-lg border border-cyan-500/30 px-3 py-1.5 text-sm text-cyan-100 hover:bg-cyan-500/10"
        >
          <LogOut size={14} /> Disconnect
        </button>
      </div>
    );
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-yellow-400 to-amber-500 px-4 py-2 text-sm font-semibold text-black hover:opacity-90"
        title="Use your own ZeroGPU quota"
      >
        <KeyRound size={16} /> Connect HF token
      </button>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <input
        type="password"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="hf_..."
        className="rounded-lg border border-cyan-500/30 bg-black/40 px-3 py-1.5 text-sm text-cyan-50 outline-none focus:border-cyan-400"
        onKeyDown={(e) => e.key === 'Enter' && connect()}
      />
      <button
        onClick={connect}
        className="rounded-lg bg-cyan-600 px-3 py-1.5 text-sm font-medium hover:bg-cyan-500"
      >
        Connect
      </button>
      <a
        href="https://huggingface.co/settings/tokens"
        target="_blank"
        rel="noreferrer"
        className="text-xs text-cyan-400 hover:underline"
      >
        get a token
      </a>
    </div>
  );
}
