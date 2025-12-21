'use client';

import { useState, useEffect } from 'react';
import {
  Sparkles,
  Zap,
  Download,
  Copy,
  Mic,
  Trash,
  ArrowRight,
  Video,
  Headphones,
  Gamepad2,
  GraduationCap,
  Megaphone,
  ShieldCheck,
  Github,
  Twitter,
  Linkedin,
  Check,
} from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_AVATAR_API_BASE || 'http://localhost:8000';

type AppState = 'idle' | 'submitting' | 'processing' | 'done' | 'error';
type StepState = '' | 'active' | 'done';

const AVATARS = [
  { id: 'professional', name: 'Professional', desc: 'Support • Sales • HR', img: 'https://picsum.photos/seed/pro/480/320' },
  { id: 'creator', name: 'Creator', desc: 'Influencer • Ads', img: 'https://picsum.photos/seed/creator/480/320' },
  { id: 'educator', name: 'Educator', desc: 'Courses • Tutors', img: 'https://picsum.photos/seed/edu/480/320' },
  { id: 'npc', name: 'Game NPC', desc: 'Dialogue • Lore', img: 'https://picsum.photos/seed/npc/480/320' },
  { id: 'brand', name: 'Brand', desc: 'Retail • Events', img: 'https://picsum.photos/seed/brand/480/320' },
  { id: 'custom', name: 'Custom', desc: 'Bring your own', img: 'https://picsum.photos/seed/custom/480/320' },
];

const USE_CASES = [
  {
    icon: Headphones,
    title: 'Customer Support',
    desc: 'A branded agent that answers FAQs and escalates to humans.',
    color: 'from-cyan-600 to-blue-600',
    textColor: 'text-cyan-300',
  },
  {
    icon: Gamepad2,
    title: 'Games & NPCs',
    desc: 'Lore-aware NPC previews for marketing or in-game cutscenes.',
    color: 'from-emerald-600 to-green-600',
    textColor: 'text-emerald-300',
  },
  {
    icon: GraduationCap,
    title: 'Education',
    desc: 'Lesson summaries, micro‑training, and multilingual tutors.',
    color: 'from-purple-600 to-pink-600',
    textColor: 'text-purple-300',
  },
  {
    icon: Megaphone,
    title: 'Marketing',
    desc: 'Product explainers, launch videos, and personalized ads.',
    color: 'from-orange-600 to-red-600',
    textColor: 'text-orange-300',
  },
];

export default function Page() {
  const [selectedAvatar, setSelectedAvatar] = useState('professional');
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [script, setScript] = useState('');
  const [qualityMode, setQualityMode] = useState('auto');

  const [jobId, setJobId] = useState<string | null>(null);
  const [state, setState] = useState<AppState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('Initializing…');
  const [currentStep, setCurrentStep] = useState(1);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.12 }
    );

    document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const offset = 86;
      const bodyRect = document.body.getBoundingClientRect().top;
      const elementRect = element.getBoundingClientRect().top;
      const elementPosition = elementRect - bodyRect;
      const offsetPosition = elementPosition - offset;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth',
      });
    }
  };

  const handleAvatarSelect = (id: string) => {
    setSelectedAvatar(id);
    setCurrentStep(2);
  };

  const handleSampleScript = () => {
    setScript(
      "Hi! I'm a realtime-ready AI avatar. Integrate me in your React UI and deploy to Vercel with a clean API route."
    );
    setCurrentStep(2);
  };

  const handleMockRecord = () => {
    setScript(
      "(Voice recorded) Welcome to the Avatar Renderer MCP demo. Watch how quickly we generate a talking head video."
    );
    setCurrentStep(2);
  };

  const handleClear = () => {
    setScript('');
    setCurrentStep(1);
  };

  const pollUntilDone = async (id: string) => {
    const steps = [
      { p: 18, s: 'Validating inputs…' },
      { p: 42, s: 'Synthesizing voice…' },
      { p: 68, s: 'Animating face rig…' },
      { p: 86, s: 'Rendering MP4…' },
      { p: 100, s: 'Finalizing…' },
    ];

    let stepIndex = 0;
    const stepInterval = setInterval(() => {
      if (stepIndex < steps.length) {
        const step = steps[stepIndex];
        setProgress(step.p);
        setStatusText(step.s);
        stepIndex++;
      }
    }, 850);

    for (let i = 0; i < 300; i++) {
      const r = await fetch(`${API_BASE}/status/${id}`);

      const ct = r.headers.get('content-type') || '';
      if (r.ok && ct.includes('video/mp4')) {
        clearInterval(stepInterval);
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        setVideoUrl(url);
        setState('done');
        setCurrentStep(4);
        return;
      }

      if (r.ok && ct.includes('application/json')) {
        // Still processing
      }

      await new Promise((resolve) => setTimeout(resolve, 1500));
    }

    clearInterval(stepInterval);
    setState('error');
    setError('Timed out waiting for the render to finish.');
  };

  const handleGenerate = async () => {
    setError(null);
    setVideoUrl(null);

    if (!avatarFile && !audioFile && !script.trim()) {
      setError('Please select an avatar image and provide audio or script.');
      return;
    }

    setState('submitting');
    setCurrentStep(3);

    const fd = new FormData();

    // For demo: use placeholder files if not provided
    if (avatarFile) {
      fd.append('avatar', avatarFile);
    } else {
      // In production, you'd require actual files
      setError('Please upload an avatar image');
      setState('error');
      return;
    }

    if (audioFile) {
      fd.append('audio', audioFile);
    } else {
      // In production, you'd require actual audio
      setError('Please upload an audio file');
      setState('error');
      return;
    }

    fd.append('qualityMode', qualityMode);

    try {
      const res = await fetch(`${API_BASE}/render-upload`, {
        method: 'POST',
        body: fd,
      });

      if (!res.ok) {
        setState('error');
        setError(`Upload failed: ${res.status} ${res.statusText}`);
        return;
      }

      const data = await res.json();
      setJobId(data.jobId);
      setState('processing');

      await pollUntilDone(data.jobId);
    } catch (err) {
      setState('error');
      setError(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const handleReset = () => {
    setState('idle');
    setVideoUrl(null);
    setJobId(null);
    setError(null);
    setProgress(0);
    setStatusText('Initializing…');
    setCurrentStep(1);
  };

  const handleDownload = () => {
    if (videoUrl) {
      window.open(videoUrl, '_blank');
    }
  };

  const handleCopyEmbed = async () => {
    const embedCode = `export default function AvatarEmbed() {\n  return (\n    <video className="rounded-2xl w-full" src="${videoUrl || 'YOUR_VIDEO_URL'}" autoPlay loop muted playsInline controls />\n  );\n}`;

    try {
      await navigator.clipboard.writeText(embedCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const getDotState = (step: number): StepState => {
    if (step < currentStep) return 'done';
    if (step === currentStep) return 'active';
    return '';
  };

  return (
    <div className="min-h-screen">
      {/* NAV */}
      <nav className="fixed w-full z-50 py-4 px-6 glass border-b border-cyan-500/10">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500"></div>
            <div>
              <div className="text-lg font-extrabold leading-none bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">
                Avatar Renderer MCP
              </div>
              <div className="text-xs text-gray-400">Vercel-ready demo • React embed</div>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-8 text-sm">
            <button onClick={() => scrollToSection('hero')} className="text-gray-300 hover:text-cyan-300 transition">
              Home
            </button>
            <button onClick={() => scrollToSection('wizard')} className="text-gray-300 hover:text-cyan-300 transition">
              Wizard
            </button>
            <button onClick={() => scrollToSection('reveal')} className="text-gray-300 hover:text-cyan-300 transition">
              Reveal
            </button>
            <button onClick={() => scrollToSection('usecases')} className="text-gray-300 hover:text-cyan-300 transition">
              Use Cases
            </button>
            <button onClick={() => scrollToSection('dev')} className="text-gray-300 hover:text-cyan-300 transition">
              Dev
            </button>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => scrollToSection('wizard')}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 font-semibold hover:from-cyan-500 hover:to-blue-500 transition glow-effect"
            >
              Launch Demo
            </button>
          </div>
        </div>
      </nav>

      {/* HERO */}
      <section id="hero" className="relative pt-28 md:pt-32 pb-16 overflow-hidden">
        <div className="container mx-auto px-6 relative z-10">
          <div className="grid lg:grid-cols-2 gap-10 items-center">
            <div className="fade-in">
              <div className="badge mb-6">
                <Sparkles className="w-4 h-4" />
                <span>Futuristic demo that sells the "wow" in 10 seconds</span>
              </div>

              <h1 className="text-5xl md:text-6xl font-extrabold leading-tight">
                <span className="bg-gradient-to-r from-cyan-300 via-blue-300 to-purple-300 bg-clip-text text-transparent">
                  Deploy a Talking Avatar
                </span>
                <br />
                <span className="text-white/95">in your React + Vercel app</span>
              </h1>

              <p className="mt-6 text-lg md:text-xl text-gray-300 max-w-xl">
                A polished "choose → speak → generate → embed" flow. Perfect for product demos, landing pages, and
                developer portals.
              </p>

              <div className="mt-8 flex flex-wrap items-center gap-3">
                <button
                  onClick={() => scrollToSection('wizard')}
                  className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 font-semibold hover:from-cyan-500 hover:to-blue-500 transition glow-effect flex items-center gap-2"
                >
                  Try the Wizard
                  <ArrowRight className="w-5 h-5" />
                </button>
                <button
                  onClick={() => scrollToSection('dev')}
                  className="px-6 py-3 rounded-xl border border-white/10 bg-white/5 text-gray-200 hover:bg-white/8 transition"
                >
                  See React Embed
                  <span className="ml-2 text-xs bg-white/10 px-2 py-1 rounded">⌘K</span>
                </button>
              </div>

              <div className="mt-10 flex flex-wrap gap-3">
                <span className="px-3 py-2 border border-white/10 bg-white/5 rounded-full text-sm">⚡ Fast "wow" moment</span>
                <span className="px-3 py-2 border border-white/10 bg-white/5 rounded-full text-sm">🧩 Embeddable component</span>
                <span className="px-3 py-2 border border-white/10 bg-white/5 rounded-full text-sm">🔒 Serverless-friendly</span>
              </div>
            </div>

            <div className="fade-in">
              <div className="neumorphic-dark p-5 md:p-6 rounded-[26px]">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-sm text-gray-300 font-semibold">Live Preview</div>
                  <div className="text-xs text-gray-400">(demo UI — wire to MCP endpoints)</div>
                </div>

                <div className="rounded-2xl overflow-hidden border border-cyan-500/15">
                  <div className="aspect-video bg-gradient-to-br from-gray-900 to-black flex items-center justify-center">
                    <div className="text-center px-6">
                      <Video className="w-16 h-16 text-white/20 mx-auto mb-3" />
                      <p className="text-gray-300 font-medium">Avatar Video Preview</p>
                      <p className="text-gray-500 text-sm mt-1">Drop in your renderer output URL here</p>
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-3">
                  <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                    <div className="text-xs text-gray-400">Latency</div>
                    <div className="text-lg font-bold text-white">~60s</div>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                    <div className="text-xs text-gray-400">Output</div>
                    <div className="text-lg font-bold text-white">MP4</div>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                    <div className="text-xs text-gray-400">Embed</div>
                    <div className="text-lg font-bold text-white">React</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/45 to-transparent my-20 mx-auto max-w-5xl"></div>

      {/* WIZARD */}
      <section id="wizard" className="py-10 md:py-14 relative">
        <div className="container mx-auto px-6">
          <div className="text-center mb-12 fade-in">
            <h2 className="text-4xl md:text-5xl font-extrabold">
              <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">
                Interactive Wizard
              </span>
            </h2>
            <p className="mt-4 text-lg text-gray-300 max-w-3xl mx-auto">
              This is the "professional demo" flow for Vercel: it guides a user from selection to generation and then
              gives them the embed snippet.
            </p>
          </div>

          {/* Steps header */}
          <div className="fade-in max-w-5xl mx-auto mb-6 neumorphic-dark p-5 md:p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
              <div>
                <div className="text-sm text-gray-400">Wizard Progress</div>
                <div className="mt-1 text-xl font-bold">Choose • Voice • Generate • Embed</div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className={`step-dot ${getDotState(1)}`}></div>
                  <div className={`step-dot ${getDotState(2)}`}></div>
                  <div className={`step-dot ${getDotState(3)}`}></div>
                  <div className={`step-dot ${getDotState(4)}`}></div>
                </div>
                <div className="text-sm text-gray-300">Step {currentStep} of 4</div>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* Left: Inputs */}
            <div className="fade-in neumorphic-dark p-6 md:p-7">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold">1) Choose an Avatar</h3>
                <span className="text-xs text-gray-400">(or upload your own)</span>
              </div>

              <div className="mt-5 grid grid-cols-2 md:grid-cols-3 gap-4">
                {AVATARS.map((avatar) => (
                  <div
                    key={avatar.id}
                    className={`avatar-tile ${selectedAvatar === avatar.id ? 'selected' : ''}`}
                    onClick={() => handleAvatarSelect(avatar.id)}
                  >
                    <img className="w-full h-28 md:h-32 object-cover" src={avatar.img} alt={avatar.name} />
                    <div className="p-3">
                      <div className="font-semibold">{avatar.name}</div>
                      <div className="text-xs text-gray-400">{avatar.desc}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6">
                <label className="block text-sm text-gray-400 mb-2">Or upload your own avatar image (PNG/JPG)</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setAvatarFile(e.target.files?.[0] || null)}
                  className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 file:cursor-pointer"
                />
              </div>

              <div className="mt-7">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-bold">2) Add Script / Audio</h3>
                  <span className="text-xs text-gray-400">(text or upload)</span>
                </div>

                <textarea
                  value={script}
                  onChange={(e) => setScript(e.target.value)}
                  className="mt-4 w-full h-36 bg-black/30 border border-cyan-500/20 rounded-2xl p-4 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/15"
                  placeholder="Type something for the avatar to say..."
                />

                <div className="mt-4">
                  <label className="block text-sm text-gray-400 mb-2">Or upload audio (WAV/MP3)</label>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                    className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 file:cursor-pointer"
                  />
                </div>

                <div className="mt-4 flex flex-wrap gap-3">
                  <button
                    onClick={handleSampleScript}
                    className="px-5 py-2.5 rounded-xl bg-white/6 border border-white/10 hover:bg-white/10 transition flex items-center gap-2"
                  >
                    <Sparkles className="w-4 h-4" />
                    Use sample
                  </button>
                  <button
                    onClick={handleMockRecord}
                    className="px-5 py-2.5 rounded-xl bg-white/6 border border-white/10 hover:bg-white/10 transition flex items-center gap-2"
                  >
                    <Mic className="w-4 h-4" />
                    Mock voice record
                  </button>
                  <button
                    onClick={handleClear}
                    className="px-5 py-2.5 rounded-xl bg-white/6 border border-white/10 hover:bg-white/10 transition flex items-center gap-2"
                  >
                    <Trash className="w-4 h-4" />
                    Clear
                  </button>
                </div>

                <div className="mt-4">
                  <label className="block text-sm text-gray-400 mb-2">Quality Mode</label>
                  <select
                    value={qualityMode}
                    onChange={(e) => setQualityMode(e.target.value)}
                    className="w-full bg-black/30 border border-cyan-500/20 rounded-xl p-3 text-white focus:outline-none focus:border-cyan-400"
                  >
                    <option value="auto">Auto</option>
                    <option value="real_time">Real-time</option>
                    <option value="high_quality">High quality</option>
                  </select>
                </div>
              </div>

              <div className="mt-8 flex flex-col md:flex-row gap-3">
                <button
                  onClick={handleGenerate}
                  disabled={state === 'submitting' || state === 'processing'}
                  className="flex-1 px-6 py-3.5 rounded-2xl bg-gradient-to-r from-emerald-600 to-cyan-600 font-bold hover:from-emerald-500 hover:to-cyan-500 transition glow-effect flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Zap className="w-5 h-5" />
                  {state === 'submitting' ? 'Uploading...' : state === 'processing' ? 'Rendering...' : 'Generate Avatar'}
                </button>
                <button
                  onClick={handleReset}
                  className="px-6 py-3.5 rounded-2xl bg-white/6 border border-white/12 hover:bg-white/10 transition"
                >
                  Reset
                </button>
              </div>

              {error && <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-300">{error}</div>}

              <div className="mt-4 text-xs text-gray-500">
                In production: the Generate button calls your MCP server → returns a video URL + metadata. This UI will
                then reveal the player + embed snippet.
              </div>
            </div>

            {/* Right: Reveal + Embed */}
            <div className="fade-in neumorphic-dark p-6 md:p-7" id="reveal">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold">3) Reveal</h3>
                <span className="text-xs text-gray-400">Autoplay preview</span>
              </div>

              <div className="mt-5 rounded-2xl overflow-hidden border border-cyan-500/15 bg-black/30">
                <div className="aspect-video relative">
                  {state === 'processing' && (
                    <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
                      <div className="text-center">
                        <div className="relative w-28 h-28 mx-auto mb-5">
                          <svg className="progress-ring w-28 h-28" viewBox="0 0 100 100">
                            <circle
                              className="stroke-white/20"
                              strokeWidth="8"
                              fill="transparent"
                              r="45"
                              cx="50"
                              cy="50"
                            ></circle>
                            <circle
                              className="progress-ring-circle stroke-emerald-400"
                              strokeWidth="8"
                              fill="transparent"
                              r="45"
                              cx="50"
                              cy="50"
                              style={{ strokeDashoffset: 283 - (progress / 100) * 283 }}
                            ></circle>
                          </svg>
                          <div className="absolute inset-0 flex items-center justify-center text-xl font-extrabold">
                            {progress}%
                          </div>
                        </div>
                        <div className="text-cyan-200 font-semibold">{statusText}</div>
                        <div className="text-xs text-gray-400 mt-2">Simulated generation (wire to MCP)</div>
                      </div>
                    </div>
                  )}

                  {!videoUrl && state !== 'processing' && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center px-8">
                        <Video className="w-16 h-16 text-white/20 mx-auto mb-3" />
                        <div className="text-gray-300 font-semibold">Generated video appears here</div>
                        <div className="text-gray-500 text-sm mt-1">Pick an avatar, add audio/script, click Generate.</div>
                      </div>
                    </div>
                  )}

                  {videoUrl && (
                    <video
                      src={videoUrl}
                      className="absolute inset-0 w-full h-full object-cover"
                      playsInline
                      muted
                      loop
                      controls
                      autoPlay
                    ></video>
                  )}
                </div>
              </div>

              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  onClick={handleDownload}
                  disabled={!videoUrl}
                  className="px-5 py-2.5 rounded-xl bg-white/6 border border-white/10 hover:bg-white/10 transition flex items-center gap-2 disabled:opacity-40"
                >
                  <Download className="w-4 h-4" />
                  Download MP4
                </button>
                <button
                  onClick={handleCopyEmbed}
                  disabled={!videoUrl}
                  className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-700 to-blue-700 hover:from-cyan-600 hover:to-blue-600 transition glow-effect flex items-center gap-2 disabled:opacity-40"
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  {copied ? 'Copied!' : 'Copy embed snippet'}
                </button>
              </div>

              <div className="mt-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-bold">4) Embed</h3>
                  <span className="text-xs text-gray-400">React / Next.js</span>
                </div>

                <div className="mt-4 bg-black/50 rounded-2xl border border-cyan-500/15 overflow-hidden">
                  <div className="flex items-center gap-2 p-3 bg-white/5 border-b border-white/10">
                    <div className="w-3 h-3 rounded-full bg-red-400"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                    <div className="w-3 h-3 rounded-full bg-green-400"></div>
                    <div className="text-xs text-gray-400 ml-2">AvatarEmbed.tsx</div>
                  </div>
                  <pre className="p-4 text-sm font-mono text-gray-300 overflow-x-auto">
                    <span className="text-cyan-400">export</span> <span className="text-cyan-400">default</span>{' '}
                    <span className="text-cyan-400">function</span> <span className="text-purple-400">AvatarEmbed</span>
                    () {'{'}
                    {'\n'}
                    {'  '}
                    <span className="text-cyan-400">return</span> ({'\n'}
                    {'    '}
                    <span className="text-gray-500">&lt;video className=&quot;rounded-2xl w-full&quot; src=&quot;</span>
                    <span className="text-green-400">{videoUrl || 'YOUR_VIDEO_URL'}</span>
                    <span className="text-gray-500">
                      &quot; autoPlay loop muted playsInline controls /&gt;
                    </span>
                    {'\n'}
                    {'  '});{'\n'}
                    {'}'}
                  </pre>
                </div>

                <div className="mt-4 text-xs text-gray-500">
                  In production: replace the placeholder URL with the video URL returned by your MCP tool. You can also
                  render a WebRTC stream instead of an MP4.
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/45 to-transparent my-20 mx-auto max-w-5xl"></div>

      {/* USE CASES */}
      <section id="usecases" className="py-10 md:py-14">
        <div className="container mx-auto px-6">
          <div className="text-center mb-12 fade-in">
            <h2 className="text-4xl md:text-5xl font-extrabold">
              <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">Use Cases</span>
            </h2>
            <p className="mt-4 text-lg text-gray-300 max-w-3xl mx-auto">
              This section sells the "potentiality" — show stakeholders how the same avatar output plugs into real
              business flows.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {USE_CASES.map((useCase, idx) => (
              <div key={idx} className="card-hover neumorphic-dark p-6">
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-r ${useCase.color} flex items-center justify-center mb-5`}>
                  <useCase.icon className="w-7 h-7" />
                </div>
                <div className="text-lg font-bold">{useCase.title}</div>
                <p className="mt-2 text-gray-400">{useCase.desc}</p>
                <div className={`mt-4 text-sm ${useCase.textColor} flex items-center gap-2`}>
                  See flow <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/45 to-transparent my-20 mx-auto max-w-5xl"></div>

      {/* DEV */}
      <section id="dev" className="py-10 md:py-14">
        <div className="container mx-auto px-6">
          <div className="text-center mb-12 fade-in">
            <h2 className="text-4xl md:text-5xl font-extrabold">
              <span className="bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">
                Developer Story
              </span>
            </h2>
            <p className="mt-4 text-lg text-gray-300 max-w-3xl mx-auto">
              Make it obvious how easy integration is: the "MCP server" runs generation, and your Next.js app just
              renders the returned URL.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
            <div className="fade-in bg-black/50 rounded-2xl border border-cyan-500/15 overflow-hidden">
              <div className="flex items-center gap-2 p-3 bg-white/5 border-b border-white/10">
                <div className="w-3 h-3 rounded-full bg-red-400"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                <div className="w-3 h-3 rounded-full bg-green-400"></div>
                <div className="text-xs text-gray-400 ml-2">1) Call MCP tool (pseudo)</div>
              </div>
              <pre className="p-4 text-sm font-mono text-gray-300">
                <span className="text-gray-500">// server action / route handler</span>
                {'\n'}
                <span className="text-cyan-400">const</span> result = <span className="text-cyan-400">await</span>{' '}
                mcp.call(<span className="text-green-400">&quot;avatar.render&quot;</span>, {'{'}
                {'\n'}
                {'  '}avatarId: <span className="text-green-400">&quot;professional&quot;</span>,{'\n'}
                {'  '}text: <span className="text-green-400">&quot;Hello from Vercel.&quot;</span>,{'\n'}
                {'}'});{'\n\n'}
                <span className="text-gray-500">// result.videoUrl -&gt; store + return to client</span>
              </pre>
            </div>

            <div className="fade-in bg-black/50 rounded-2xl border border-cyan-500/15 overflow-hidden">
              <div className="flex items-center gap-2 p-3 bg-white/5 border-b border-white/10">
                <div className="w-3 h-3 rounded-full bg-red-400"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                <div className="w-3 h-3 rounded-full bg-green-400"></div>
                <div className="text-xs text-gray-400 ml-2">2) Render in React</div>
              </div>
              <pre className="p-4 text-sm font-mono text-gray-300">
                <span className="text-cyan-400">export</span> <span className="text-cyan-400">function</span>{' '}
                <span className="text-purple-400">AvatarPlayer</span>({'{'}videoUrl{'}'}) {'{'}
                {'\n'}
                {'  '}
                <span className="text-cyan-400">return</span> ({'\n'}
                {'    '}
                <span className="text-gray-500">&lt;video src=&quot;</span>
                <span className="text-green-400">{'{'}videoUrl{'}'}</span>
                <span className="text-gray-500">&quot; autoPlay loop muted playsInline controls /&gt;</span>
                {'\n'}
                {'  '});{'\n'}
                {'}'}
              </pre>
            </div>
          </div>

          <div className="max-w-5xl mx-auto mt-8 fade-in neumorphic-dark p-6 md:p-7">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-r from-cyan-600 to-blue-600 flex items-center justify-center flex-shrink-0">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <div>
                <div className="text-xl font-bold">What makes this demo &quot;professional&quot;</div>
                <ul className="mt-3 text-gray-300 space-y-2 list-disc pl-5">
                  <li>
                    <span className="font-semibold">Single-page narrative</span>: Hero → Wizard → Reveal → Use cases →
                    Dev embed.
                  </li>
                  <li>
                    <span className="font-semibold">Instant wow</span>: autoplay preview area + clean progress UI.
                  </li>
                  <li>
                    <span className="font-semibold">Clear integration</span>: copyable snippet after generation.
                  </li>
                  <li>
                    <span className="font-semibold">Vercel-friendly</span>: generation behind a route handler; frontend
                    is static + fast.
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-12 border-t border-cyan-500/10 mt-20">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div>
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500"></div>
                <div>
                  <div className="text-lg font-extrabold bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">
                    Avatar Renderer MCP
                  </div>
                  <div className="text-xs text-gray-500">© 2025 • Demo template for Vercel</div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 text-gray-400">
              <a href="#" className="hover:text-cyan-300 transition" aria-label="GitHub">
                <Github className="w-5 h-5" />
              </a>
              <a href="#" className="hover:text-cyan-300 transition" aria-label="X">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="#" className="hover:text-cyan-300 transition" aria-label="LinkedIn">
                <Linkedin className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
