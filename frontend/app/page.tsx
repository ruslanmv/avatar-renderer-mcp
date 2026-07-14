'use client';

import { useState, useEffect } from 'react';
import {
  Sparkles,
  Zap,
  Download,
  Mic,
  Play,
  RefreshCw,
  Video,
  ArrowRight,
  Headphones,
  Gamepad2,
  GraduationCap,
  Megaphone,
  Twitter,
  Github,
  Linkedin,
  Youtube,
  Cpu,
  UploadCloud,
  Trash,
} from 'lucide-react';

import HuggingFaceLogin from '../components/HuggingFaceLogin';
import HfTokenConnect from '../components/HfTokenConnect';
import { generateAvatar } from '../lib/gradioClient';
import { getHfToken } from '../lib/hfToken';

const SAMPLE_SCRIPT =
  'Welcome to Avatar Renderer MCP. Pick a face, type your message, choose a voice, and generate a polished talking-avatar video on Hugging Face powered infrastructure.';

const AVATARS = [
  {
    id: 'professional',
    name: 'Professional',
    desc: 'Support • Sales • HR',
    img: 'https://image.pollinations.ai/prompt/professional%20business%20portrait%20photography%20studio%20lighting%20neutral%20background?width=400&height=400&nologo=true',
  },
  {
    id: 'creator',
    name: 'Creator',
    desc: 'Influencer • Ads',
    img: 'https://image.pollinations.ai/prompt/casual%20portrait%20young%20person%20smiling%20natural%20lighting%20lifestyle%20photography?width=400&height=400&nologo=true',
  },
  {
    id: 'educator',
    name: 'Educator',
    desc: 'Courses • Tutors',
    img: 'https://image.pollinations.ai/prompt/teacher%20portrait%20friendly%20professional%20education%20setting?width=400&height=400&nologo=true',
  },
  {
    id: 'npc',
    name: 'Game NPC',
    desc: 'Dialogue • Lore',
    img: 'https://image.pollinations.ai/prompt/3d%20render%20game%20character%20fantasy%20rpg%20style%20detailed?width=400&height=400&nologo=true',
  },
  {
    id: 'brand',
    name: 'Brand',
    desc: 'Retail • Events',
    img: 'https://image.pollinations.ai/prompt/brand%20ambassador%20professional%20corporate%20portrait?width=400&height=400&nologo=true',
  },
  {
    id: 'custom',
    name: 'Custom',
    desc: 'Bring your own',
    img: 'https://image.pollinations.ai/prompt/futuristic%20cyberpunk%20digital%20avatar%20neon%20lights%20sci-fi%20style?width=400&height=400&nologo=true',
  },
];

export default function Page() {
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [script, setScript] = useState(SAMPLE_SCRIPT);
  const [selectedAvatar, setSelectedAvatar] = useState('professional');
  const [qualityMode, setQualityMode] = useState('auto');
  // Audio source: 'tts' = synthesize from the script text, 'upload' = use the file.
  const [audioSource, setAudioSource] = useState<'tts' | 'upload'>('tts');
  const [voice, setVoice] = useState('en-US-AriaNeural');
  const [speed, setSpeed] = useState(0);
  const [enabledEnhancements, setEnabledEnhancements] = useState<string[]>([
    'emotion_expressions',
    'eye_gaze_blink',
    'gesture_animation',
  ]);

  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('Initializing...');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  const selectedAvatarConfig = AVATARS.find((avatar) => avatar.id === selectedAvatar) ?? AVATARS[0];

  const loadSelectedAvatarFile = async (): Promise<File> => {
    const response = await fetch(selectedAvatarConfig.img);
    if (!response.ok) {
      throw new Error(`Could not load selected avatar image: ${selectedAvatarConfig.name}`);
    }
    const blob = await response.blob();
    return new File([blob], `${selectedAvatarConfig.id}.png`, { type: blob.type || 'image/png' });
  };

  const handleGenerate = async () => {
    if (isGenerating) return;
    const useText = audioSource === 'tts';
    if (useText && !script.trim()) {
      alert('Please type some text to speak (or switch to Upload audio).');
      return;
    }
    if (!useText && !audioFile) {
      alert('Please upload an audio file (or switch to Text-to-speech).');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setStatusText('Connecting to GPU backend...');
    setVideoUrl(null);

    // Indeterminate progress while the Space runs inference on ZeroGPU.
    const steps = [
      { p: 20, s: 'Uploading inputs...' },
      { p: 45, s: 'Allocating GPU...' },
      { p: 70, s: 'Rendering avatar...' },
      { p: 90, s: 'Finalizing video...' },
    ];
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
      if (stepIndex < steps.length) {
        setProgress(steps[stepIndex].p);
        setStatusText(steps[stepIndex].s);
        stepIndex++;
      }
    }, 1200);

    try {
      const imageForGeneration = avatarFile ?? await loadSelectedAvatarFile();

      // Inference runs entirely on the Hugging Face Space (no GPU on Vercel).
      // Pass the user's HF token (if connected) so the run uses THEIR ZeroGPU quota.
      const url = await generateAvatar({
        image: imageForGeneration,
        audio: useText ? null : audioFile,
        text: useText ? script.trim() : '',
        voice,
        speed,
        quality: qualityMode,
        enhancements: enabledEnhancements,
        hfToken: getHfToken() ?? undefined,
      });
      clearInterval(stepInterval);
      setVideoUrl(url);
      setProgress(100);
      setStatusText('Complete!');
    } catch (err) {
      clearInterval(stepInterval);
      alert(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCreateAnother = () => {
    setVideoUrl(null);
    setProgress(0);
    setStatusText('Initializing...');
  };

  const handleDownload = () => {
    if (videoUrl) {
      window.open(videoUrl, '_blank');
    }
  };

  const handleClear = () => {
    setScript('');
    setAudioFile(null);
    setAvatarFile(null);
  };

  const handleSample = () => {
    setScript(SAMPLE_SCRIPT);
  };

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const offset = 80;
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

  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed w-full z-50 py-4 px-6 bg-black/30 backdrop-blur-lg border-b border-cyan-500/20">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg"></div>
            <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Avatar Renderer MCP
            </span>
          </div>

          <div className="hidden md:flex space-x-8">
            <button onClick={() => scrollToSection('hero')} className="text-gray-300 hover:text-cyan-400 transition-colors">
              Home
            </button>
            <button onClick={() => scrollToSection('demo')} className="text-gray-300 hover:text-cyan-400 transition-colors">
              Demo
            </button>
            <button onClick={() => scrollToSection('reveal')} className="text-gray-300 hover:text-cyan-400 transition-colors">
              Reveal
            </button>
            <button onClick={() => scrollToSection('usecases')} className="text-gray-300 hover:text-cyan-400 transition-colors">
              Use Cases
            </button>
            <button onClick={() => scrollToSection('tech')} className="text-gray-300 hover:text-cyan-400 transition-colors">
              Tech
            </button>
          </div>

          <div className="flex items-center gap-4">
            <HfTokenConnect />
            <HuggingFaceLogin />
            <button
              onClick={() => scrollToSection('demo')}
              className="px-6 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-lg font-medium hover:from-cyan-500 hover:to-blue-500 transition-all glow-effect"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="hero" className="min-h-screen flex items-center justify-center relative overflow-hidden pt-20">
        <div className="container mx-auto px-6 text-center z-10">
          <div className="fade-in mb-8">
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                Sell, Teach, and Support
                <br />
                with AI Avatars
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto mb-12">
              Turn one image and one script into a polished talking-head video for product demos, training, support, and creator content. Run generation through Hugging Face powered GPU infrastructure—no GPU on Vercel required.
            </p>
          </div>

          <div className="fade-in mb-12">
            <div className="relative max-w-4xl mx-auto">
              <div className="relative rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,150,255,0.3)]">
                <video
                  className="aspect-video w-full object-cover bg-black"
                  src="/demo.mp4"
                  poster="/demo-poster.jpg"
                  autoPlay
                  loop
                  muted
                  playsInline
                  controls
                  preload="metadata"
                />
                <div className="absolute inset-0 border border-cyan-500/30 rounded-2xl pointer-events-none"></div>
              </div>

              <div className="absolute -top-4 -left-4 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl"></div>
              <div className="absolute -bottom-4 -right-4 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl"></div>
            </div>
          </div>

          <div className="fade-in mb-8 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl mx-auto text-left">
            {[
              ['1', 'Pick a face', 'Use a sample avatar or upload your own image.'],
              ['2', 'Type a pitch', 'Use text-to-speech or upload finished audio.'],
              ['3', 'Render on HF', 'Send the job to a Hugging Face Space and download MP4.'],
            ].map(([step, title, desc]) => (
              <div key={step} className="neumorphic-dark rounded-2xl p-5 border border-cyan-500/20">
                <div className="text-cyan-300 text-sm font-bold mb-2">Step {step}</div>
                <div className="text-lg font-semibold mb-1">{title}</div>
                <div className="text-sm text-gray-400">{desc}</div>
              </div>
            ))}
          </div>

          <div className="fade-in">
            <button
              onClick={() => scrollToSection('demo')}
              className="px-8 py-4 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-xl font-semibold text-lg hover:from-cyan-500 hover:to-blue-500 transition-all transform hover:scale-105 glow-effect"
            >
              Start Creating →
            </button>
          </div>
        </div>

        <div className="absolute top-1/4 left-10 w-4 h-4 bg-cyan-400 rounded-full blur-sm opacity-50"></div>
        <div className="absolute bottom-1/4 right-10 w-6 h-6 bg-blue-400 rounded-full blur-sm opacity-30"></div>
      </section>

      {/* Interactive Demo Section */}
      <section id="demo" className="py-20 relative">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Interactive Demo
              </span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Select a sample avatar or upload your own image, type a commercial script, and render through the Hugging Face backend.
            </p>
          </div>

          {/* Step 1: Avatar Selection */}
          <div className="mb-20 fade-in">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-full flex items-center justify-center mr-4">
                  <span className="text-xl font-bold">1</span>
                </div>
                <h3 className="text-2xl font-semibold">Choose Your Avatar</h3>
              </div>
              <span className="text-sm text-gray-400">(or upload your own)</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
              {AVATARS.map((avatar) => (
                <div
                  key={avatar.id}
                  onClick={() => {
                    setSelectedAvatar(avatar.id);
                    setAvatarFile(null);
                  }}
                  className={`neumorphic-dark rounded-xl p-3 cursor-pointer transition-all ${
                    selectedAvatar === avatar.id
                      ? 'ring-2 ring-cyan-500 shadow-[0_0_30px_rgba(0,150,255,0.5)]'
                      : 'hover:scale-105 hover:shadow-[0_0_30px_rgba(0,150,255,0.3)]'
                  }`}
                >
                  <img
                    src={avatar.img}
                    alt={avatar.name}
                    className="w-full h-32 object-cover rounded-lg mb-3"
                    loading="lazy"
                  />
                  <p className="text-center font-medium text-sm">{avatar.name}</p>
                  <p className="text-center text-xs text-gray-400">{avatar.desc}</p>
                </div>
              ))}
            </div>

            {/* Upload Section */}
            <div className="mt-8">
              <div className="relative py-4">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-cyan-500/30"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-[#0a0a2a] text-gray-400">Selected samples work immediately, or upload your own avatar image (PNG/JPG)</span>
                </div>
              </div>

              <div className="mt-6">
                <label className="flex flex-col items-center justify-center w-full max-w-2xl mx-auto h-32 border-2 border-dashed border-cyan-500/30 rounded-xl cursor-pointer hover:border-cyan-400 hover:bg-cyan-500/5 transition-all group">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <UploadCloud className="w-8 h-8 text-cyan-400 mb-2 group-hover:scale-110 transition-transform" />
                    <p className="text-sm text-gray-300">
                      {avatarFile ? avatarFile.name : `Using sample: ${selectedAvatarConfig.name} (click to upload your own)`}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">PNG, JPG up to 10MB</p>
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    accept="image/png, image/jpeg"
                    onChange={(e) => setAvatarFile(e.target.files?.[0] || null)}
                  />
                </label>
              </div>
            </div>
          </div>

          {/* Step 2: Script/Audio Input */}
          <div className="mb-20 fade-in">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-full flex items-center justify-center mr-4">
                  <span className="text-xl font-bold">2</span>
                </div>
                <h3 className="text-2xl font-semibold">Add Script / Audio</h3>
              </div>
              <span className="text-sm text-gray-400">(text or upload)</span>
            </div>

            <div className="neumorphic-dark rounded-2xl p-8 max-w-3xl mx-auto">
              <div className="mb-6">
                <textarea
                  value={script}
                  onChange={(e) => setScript(e.target.value)}
                  className="w-full h-32 bg-gray-900/50 border border-cyan-500/30 rounded-xl p-4 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
                  placeholder="Type something for the avatar to say..."
                />
              </div>

              <div className="mb-6">
                <label className="block text-sm text-gray-400 mb-2">Or upload audio (WAV/MP3)</label>
                <div className="relative">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                    className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 file:cursor-pointer"
                  />
                </div>
                {audioFile && (
                  <p className="text-xs text-cyan-400 mt-2">Selected: {audioFile.name}</p>
                )}
              </div>

              <div className="mb-6">
                <label className="block text-sm text-gray-400 mb-2">Quality Mode</label>
                <select
                  value={qualityMode}
                  onChange={(e) => setQualityMode(e.target.value)}
                  className="w-full bg-gray-900/50 border border-cyan-500/30 rounded-xl p-3 text-white focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
                >
                  <option value="auto">Auto</option>
                  <option value="real_time">Real-time</option>
                  <option value="high_quality">High quality</option>
                </select>
              </div>

              {/* Audio source + Text-to-Speech settings */}
              <div className="mb-6">
                <label className="block text-sm text-gray-400 mb-3">Audio source</label>
                <div className="flex gap-2 mb-3">
                  {([['tts', 'Text-to-speech'], ['upload', 'Upload audio']] as const).map(([val, lbl]) => (
                    <button
                      key={val}
                      type="button"
                      onClick={() => setAudioSource(val)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                        audioSource === val
                          ? 'border-cyan-500 bg-cyan-500/20 text-cyan-300'
                          : 'border-gray-600 text-gray-500 hover:opacity-80'
                      }`}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>
                {audioSource === 'tts' && (
                  <div className="grid grid-cols-1 gap-3">
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Voice</label>
                      <select
                        value={voice}
                        onChange={(e) => setVoice(e.target.value)}
                        className="w-full bg-black/40 border border-gray-600 rounded-lg px-3 py-2 text-sm text-cyan-50 outline-none focus:border-cyan-400"
                      >
                        {[
                          ['en-US-AriaNeural', 'English (US) — Aria, female'],
                          ['en-US-GuyNeural', 'English (US) — Guy, male'],
                          ['en-GB-SoniaNeural', 'English (UK) — Sonia, female'],
                          ['en-GB-RyanNeural', 'English (UK) — Ryan, male'],
                          ['es-ES-ElviraNeural', 'Spanish (ES) — Elvira, female'],
                          ['es-MX-JorgeNeural', 'Spanish (MX) — Jorge, male'],
                          ['fr-FR-DeniseNeural', 'French — Denise, female'],
                          ['de-DE-KatjaNeural', 'German — Katja, female'],
                          ['it-IT-ElsaNeural', 'Italian — Elsa, female'],
                          ['pt-BR-FranciscaNeural', 'Portuguese (BR) — Francisca, female'],
                          ['hi-IN-SwaraNeural', 'Hindi — Swara, female'],
                          ['zh-CN-XiaoxiaoNeural', 'Chinese — Xiaoxiao, female'],
                          ['ja-JP-NanamiNeural', 'Japanese — Nanami, female'],
                          ['ar-EG-SalmaNeural', 'Arabic — Salma, female'],
                          ['ru-RU-SvetlanaNeural', 'Russian — Svetlana, female'],
                        ].map(([id, label]) => (
                          <option key={id} value={id}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Speed: {speed > 0 ? `+${speed}` : speed}%</label>
                      <input
                        type="range"
                        min={-50}
                        max={50}
                        step={5}
                        value={speed}
                        onChange={(e) => setSpeed(parseInt(e.target.value, 10))}
                        className="w-full accent-cyan-500"
                      />
                    </div>
                    <p className="text-xs text-gray-500">
                      The text in the script box above is spoken in the selected voice.
                    </p>
                  </div>
                )}
              </div>

              {/* Enhancement Toggles */}
              <div className="mb-6">
                <label className="block text-sm text-gray-400 mb-3">Enhancements</label>
                <div className="flex flex-wrap gap-2">
                  {[
                    { id: 'emotion_expressions', label: 'Emotion', color: 'cyan', ready: true },
                    { id: 'eye_gaze_blink', label: 'Eye Gaze', color: 'purple', ready: true },
                    { id: 'gesture_animation', label: 'Gestures', color: 'green', ready: true },
                    { id: 'musetalk_lipsync', label: 'MuseTalk', color: 'blue', ready: false },
                    { id: 'liveportrait_driver', label: 'LivePortrait', color: 'emerald', ready: false },
                    { id: 'latentsync_lipsync', label: 'LatentSync', color: 'amber', ready: false },
                    { id: 'hallo3_cinematic', label: 'Hallo3', color: 'red', ready: false },
                    { id: 'cosyvoice_tts', label: 'CosyVoice', color: 'sky', ready: false },
                    { id: 'viseme_guided', label: 'Viseme', color: 'pink', ready: false },
                    { id: 'gaussian_splatting', label: '3D Gauss', color: 'orange', ready: false },
                  ].map((enh) => {
                    const isActive = enabledEnhancements.includes(enh.id);
                    return (
                      <button
                        key={enh.id}
                        type="button"
                        onClick={() => {
                          setEnabledEnhancements((prev) =>
                            prev.includes(enh.id) ? prev.filter((e) => e !== enh.id) : [...prev, enh.id]
                          );
                        }}
                        className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${
                          isActive
                            ? `border-${enh.color}-500 bg-${enh.color}-500/20 text-${enh.color}-300 shadow-[0_0_8px_rgba(0,150,255,0.2)]`
                            : 'border-gray-600 text-gray-500 opacity-50 hover:opacity-75'
                        }`}
                        title={enh.ready ? enh.label : `${enh.label} (requires model download)`}
                      >
                        {enh.label}
                      </button>
                    );
                  })}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  {enabledEnhancements.length} enhancement{enabledEnhancements.length !== 1 ? 's' : ''} selected
                </p>
              </div>

              <div className="flex flex-wrap gap-4 justify-center">
                <button
                  onClick={handleSample}
                  className="px-6 py-3 bg-gradient-to-r from-cyan-700 to-cyan-800 rounded-lg font-medium hover:from-cyan-600 hover:to-cyan-700 transition-all flex items-center glow-effect"
                >
                  <Sparkles className="w-5 h-5 mr-2" />
                  Use sample
                </button>
                <button className="px-6 py-3 bg-gradient-to-r from-blue-700 to-blue-800 rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 transition-all flex items-center glow-effect">
                  <Mic className="w-5 h-5 mr-2" />
                  Mock voice record
                </button>
                <button
                  onClick={handleClear}
                  className="px-6 py-3 bg-gradient-to-r from-purple-700 to-purple-800 rounded-lg font-medium hover:from-purple-600 hover:to-purple-700 transition-all flex items-center glow-effect"
                >
                  <Trash className="w-5 h-5 mr-2" />
                  Clear
                </button>
              </div>
            </div>
          </div>

          {/* Step 3: Generate */}
          <div className="fade-in">
            <div className="flex items-center mb-8">
              <div className="w-12 h-12 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-full flex items-center justify-center mr-4">
                <span className="text-xl font-bold">3</span>
              </div>
              <h3 className="text-2xl font-semibold">Generate Your Avatar</h3>
            </div>

            <div className="text-center">
              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className={`px-12 py-6 rounded-2xl font-bold text-xl transition-all transform hover:scale-105 glow-effect ${
                  isGenerating
                    ? 'bg-gradient-to-r from-gray-700 to-gray-800 cursor-not-allowed'
                    : 'bg-gradient-to-r from-green-600 to-cyan-600 hover:from-green-500 hover:to-cyan-500'
                }`}
              >
                {isGenerating ? (
                  <span className="flex items-center gap-3">
                    <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Generating...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    Bring to Life
                    <Zap className="w-6 h-6" />
                  </span>
                )}
              </button>

              <button
                onClick={handleCreateAnother}
                className="ml-4 px-6 py-3 bg-white/6 border border-white/12 rounded-xl hover:bg-white/10 transition-all"
              >
                Reset
              </button>

              <p className="mt-4 text-sm text-gray-400">
                Using {avatarFile ? 'your uploaded image' : `the ${selectedAvatarConfig.name} sample image`} with {audioSource === 'tts' ? 'text-to-speech' : audioFile ? audioFile.name : 'uploaded audio'}.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent my-20 mx-auto max-w-4xl"></div>

      {/* Reveal Section */}
      <section id="reveal" className="py-20 relative">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Your AI Avatar
              </span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Watch your creation come to life with stunning realism
            </p>
          </div>

          <div className="max-w-4xl mx-auto fade-in">
            <div className="neumorphic-dark rounded-3xl overflow-hidden p-2 shadow-[0_0_50px_rgba(0,150,255,0.3)]">
              <div className="aspect-video bg-gradient-to-br from-gray-900 to-black rounded-2xl overflow-hidden relative">
                {/* Progress Overlay */}
                {isGenerating && (
                  <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-10">
                    <div className="text-center">
                      <div className="relative w-32 h-32 mx-auto mb-6">
                        <svg className="w-32 h-32 -rotate-90" viewBox="0 0 100 100">
                          <circle className="stroke-cyan-500" strokeWidth="8" fill="transparent" r="45" cx="50" cy="50" />
                          <circle
                            className="stroke-green-500 transition-all duration-500"
                            strokeWidth="8"
                            fill="transparent"
                            r="45"
                            cx="50"
                            cy="50"
                            strokeDasharray="283"
                            strokeDashoffset={283 - (progress / 100) * 283}
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center text-xl font-bold">
                          {progress}%
                        </div>
                      </div>
                      <p className="text-xl font-medium text-cyan-300">{statusText}</p>
                    </div>
                  </div>
                )}

                {/* Video Player */}
                {videoUrl ? (
                  <video src={videoUrl} className="w-full h-full object-cover" controls autoPlay loop muted playsInline />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="text-center">
                      <Video className="w-20 h-20 text-gray-600 mx-auto mb-4" />
                      <p className="text-gray-500 text-lg">Your generated avatar will appear here</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-wrap gap-6 justify-center mt-12">
              <button
                onClick={handleDownload}
                disabled={!videoUrl}
                className={`px-8 py-3 rounded-xl font-medium transition-all flex items-center glow-effect ${
                  videoUrl
                    ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500'
                    : 'bg-gradient-to-r from-cyan-700 to-blue-700 opacity-50 cursor-not-allowed'
                }`}
              >
                <Download className="w-5 h-5 mr-2" />
                Download MP4
              </button>
              <button
                onClick={handleCreateAnother}
                className="px-8 py-3 bg-gradient-to-r from-purple-700 to-pink-700 rounded-xl font-medium hover:from-purple-600 hover:to-pink-600 transition-all flex items-center glow-effect"
              >
                <RefreshCw className="w-5 h-5 mr-2" />
                Create Another
              </button>
            </div>
          </div>
        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent my-20 mx-auto max-w-4xl"></div>

      {/* Use Cases */}
      <section id="usecases" className="py-20">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">Use Cases</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">Transform industries with AI-powered avatars</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: Headphones,
                title: 'Customer Support',
                desc: 'Human-style onboarding, FAQs, and product support videos at scale',
                color: 'from-cyan-600 to-blue-600',
                textColor: 'text-cyan-400',
              },
              {
                icon: Gamepad2,
                title: 'Game NPCs',
                desc: 'Dynamic NPCs with unique personalities and dialogue',
                color: 'from-green-600 to-emerald-600',
                textColor: 'text-green-400',
              },
              {
                icon: GraduationCap,
                title: 'AI Teacher',
                desc: 'Personalized educational avatars that adapt to learning styles',
                color: 'from-purple-600 to-pink-600',
                textColor: 'text-purple-400',
              },
              {
                icon: Megaphone,
                title: 'Marketing',
                desc: 'Reusable product spokespeople for ads, launches, and landing pages',
                color: 'from-orange-600 to-red-600',
                textColor: 'text-orange-400',
              },
            ].map((item, idx) => (
              <div key={idx} className="neumorphic-dark rounded-2xl p-8 transition-all hover:-translate-y-2 hover:shadow-2xl">
                <div className={`w-16 h-16 bg-gradient-to-r ${item.color} rounded-xl flex items-center justify-center mb-6`}>
                  <item.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold mb-4">{item.title}</h3>
                <p className="text-gray-400 mb-6">{item.desc}</p>
                <div className={`flex items-center ${item.textColor}`}>
                  <span>Learn more</span>
                  <ArrowRight className="w-4 h-4 ml-2" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent my-20 mx-auto max-w-4xl"></div>

      {/* Tech Specs */}
      <section id="tech" className="py-20">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Technical Specifications
              </span>
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto mb-20">
            {[
              { icon: Zap, title: 'Simple Workflow', desc: 'Pick image, write text, choose voice, render', color: 'from-cyan-600 to-blue-600' },
              { icon: Video, title: 'Commercial MP4s', desc: 'Landing-page-ready videos for sales and education', color: 'from-green-600 to-emerald-600' },
              { icon: Cpu, title: 'Hugging Face Backend', desc: 'Vercel UI connects to GPU/ZeroGPU Spaces for inference', color: 'from-purple-600 to-pink-600' },
            ].map((item, idx) => (
              <div key={idx} className="text-center p-6">
                <div className={`w-12 h-12 bg-gradient-to-r ${item.color} rounded-lg flex items-center justify-center mx-auto mb-4`}>
                  <item.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-gray-400">{item.desc}</p>
              </div>
            ))}
          </div>

          <div className="text-center">
            <h2 className="text-5xl md:text-6xl font-bold mb-8">
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                Create Your First Avatar
              </span>
            </h2>
            <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
              Start with a sample face, connect Hugging Face for GPU-backed rendering, and publish avatar videos anywhere.
            </p>
            <button
              onClick={() => scrollToSection('demo')}
              className="px-12 py-6 bg-gradient-to-r from-cyan-600 to-purple-600 rounded-2xl font-bold text-xl hover:from-cyan-500 hover:to-purple-500 transition-all transform hover:scale-105 glow-effect"
            >
              Start Free Trial
              <Sparkles className="w-6 h-6 ml-2 inline" />
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-cyan-500/20">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-6 md:mb-0">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg"></div>
                <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  Avatar Renderer MCP
                </span>
              </div>
              <p className="text-gray-500 mt-2">© 2025 Avatar Renderer MCP. All rights reserved.</p>
            </div>

            <div className="flex space-x-6">
              {[Twitter, Github, Linkedin, Youtube].map((Icon, idx) => (
                <a key={idx} href="#" className="text-gray-400 hover:text-cyan-400 transition-colors">
                  <Icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </footer>

      <style jsx>{`
        @keyframes waveform {
          0%,
          100% {
            height: 5px;
          }
          50% {
            height: 25px;
          }
        }
      `}</style>
    </div>
  );
}
