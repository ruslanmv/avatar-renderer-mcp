/* ═══════════════════════════════════════════════════════════════════════════
   Avatar Renderer MCP — Enterprise Frontend Logic
   Communicates with FastAPI backend via REST API
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = '';  // Same origin (served by FastAPI)

/* ── Avatar presets ───────────────────────────────────────────────────────── */
const AVATARS = [
  { id:'professional', name:'Professional', desc:'Support / Sales / HR',
    img:'https://image.pollinations.ai/prompt/professional%20business%20portrait%20photography%20studio%20lighting%20neutral%20background?width=400&height=400&nologo=true' },
  { id:'creator', name:'Creator', desc:'Influencer / Ads',
    img:'https://image.pollinations.ai/prompt/casual%20portrait%20young%20person%20smiling%20natural%20lighting%20lifestyle%20photography?width=400&height=400&nologo=true' },
  { id:'educator', name:'Educator', desc:'Courses / Tutors',
    img:'https://image.pollinations.ai/prompt/teacher%20portrait%20friendly%20professional%20education%20setting?width=400&height=400&nologo=true' },
  { id:'npc', name:'Game NPC', desc:'Dialogue / Lore',
    img:'https://image.pollinations.ai/prompt/3d%20render%20game%20character%20fantasy%20rpg%20style%20detailed?width=400&height=400&nologo=true' },
  { id:'brand', name:'Brand', desc:'Retail / Events',
    img:'https://image.pollinations.ai/prompt/brand%20ambassador%20professional%20corporate%20portrait?width=400&height=400&nologo=true' },
  { id:'custom', name:'Custom', desc:'Bring your own',
    img:'https://image.pollinations.ai/prompt/futuristic%20cyberpunk%20digital%20avatar%20neon%20lights%20sci-fi%20style?width=400&height=400&nologo=true' },
];

const ENHANCEMENTS = [
  { id:'emotion_expressions', label:'Emotion', color:'#6366f1', checked:true },
  { id:'eye_gaze_blink',      label:'Eye Gaze', color:'#8b5cf6', checked:true },
  { id:'gesture_animation',   label:'Gestures', color:'#84cc16', checked:true },
  { id:'musetalk_lipsync',    label:'MuseTalk', color:'#3b82f6', checked:false },
  { id:'liveportrait_driver', label:'LivePortrait', color:'#10b981', checked:false },
  { id:'latentsync_lipsync',  label:'LatentSync', color:'#f59e0b', checked:false },
  { id:'hallo3_cinematic',    label:'Hallo3', color:'#ef4444', checked:false },
  { id:'cosyvoice_tts',       label:'CosyVoice', color:'#06b6d4', checked:false },
  { id:'viseme_guided',       label:'Viseme', color:'#ec4899', checked:false },
  { id:'gaussian_splatting',  label:'3D Gauss', color:'#f97316', checked:false },
];

/* ── State ────────────────────────────────────────────────────────────────── */
let selectedAvatar = 'professional';
let avatarFile = null;
let audioFile = null;
let isGenerating = false;
let videoUrl = null;

/* ── Init ─────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  renderAvatarGrid();
  renderEnhancements();
  setupEventListeners();
  checkSystemStatus();
  initFadeIn();
});

/* ── Render Avatar Grid ───────────────────────────────────────────────────── */
function renderAvatarGrid() {
  const grid = document.getElementById('avatar-grid');
  grid.innerHTML = AVATARS.map(a => `
    <div class="avatar-tile${a.id === selectedAvatar ? ' selected' : ''}" data-id="${a.id}">
      <img src="${a.img}" alt="${a.name}" loading="lazy">
      <p class="name">${a.name}</p>
      <p class="desc">${a.desc}</p>
    </div>
  `).join('');

  grid.querySelectorAll('.avatar-tile').forEach(tile => {
    tile.addEventListener('click', () => {
      selectedAvatar = tile.dataset.id;
      grid.querySelectorAll('.avatar-tile').forEach(t => t.classList.remove('selected'));
      tile.classList.add('selected');
    });
  });
}

/* ── Render Enhancement Toggles ───────────────────────────────────────────── */
function renderEnhancements() {
  const grid = document.getElementById('enh-grid');
  grid.innerHTML = ENHANCEMENTS.map(e => `
    <label class="enh-toggle" title="${e.label}">
      <input type="checkbox" name="enh" value="${e.id}" ${e.checked ? 'checked' : ''}>
      <span class="enh-chip" style="--enh-color:${e.color}">${e.label}</span>
    </label>
  `).join('');
  updateEnhCount();
  grid.querySelectorAll('input').forEach(cb => cb.addEventListener('change', updateEnhCount));
}

function updateEnhCount() {
  const n = document.querySelectorAll('[name="enh"]:checked').length;
  document.getElementById('enh-count').textContent = `${n} selected`;
}

/* ── Event Listeners ──────────────────────────────────────────────────────── */
function setupEventListeners() {
  // Avatar file upload
  const avatarInput = document.getElementById('avatar-file-input');
  const avatarZone = document.getElementById('avatar-upload-zone');
  avatarInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
      avatarFile = e.target.files[0];
      document.getElementById('avatar-upload-text').textContent = avatarFile.name;
      // Deselect preset avatars
      selectedAvatar = 'custom';
      document.querySelectorAll('.avatar-tile').forEach(t => t.classList.remove('selected'));
      const custom = document.querySelector('[data-id="custom"]');
      if (custom) custom.classList.add('selected');
    }
  });
  // Drag and drop
  avatarZone.addEventListener('dragover', e => { e.preventDefault(); avatarZone.style.borderColor = '#22d3ee'; });
  avatarZone.addEventListener('dragleave', () => { avatarZone.style.borderColor = ''; });
  avatarZone.addEventListener('drop', e => {
    e.preventDefault(); avatarZone.style.borderColor = '';
    if (e.dataTransfer.files[0]) {
      avatarFile = e.dataTransfer.files[0];
      document.getElementById('avatar-upload-text').textContent = avatarFile.name;
    }
  });

  // Audio file
  document.getElementById('audio-file-input').addEventListener('change', (e) => {
    audioFile = e.target.files[0] || null;
    document.getElementById('audio-file-label').textContent = audioFile ? `Selected: ${audioFile.name}` : '';
  });

  // Sample button
  document.getElementById('sample-btn').addEventListener('click', () => {
    document.getElementById('script-input').value =
      'Hello! Welcome to Avatar Renderer MCP demo. Watch how quickly we generate a talking head video with realistic lip synchronization.';
  });

  // Clear button
  document.getElementById('clear-btn').addEventListener('click', () => {
    document.getElementById('script-input').value = '';
    audioFile = null;
    avatarFile = null;
    document.getElementById('audio-file-input').value = '';
    document.getElementById('audio-file-label').textContent = '';
    document.getElementById('avatar-upload-text').textContent = 'Click to upload or drag and drop';
  });

  // Generate
  document.getElementById('generate-btn').addEventListener('click', handleGenerate);
  document.getElementById('reset-btn').addEventListener('click', handleReset);
  document.getElementById('create-another-btn').addEventListener('click', handleReset);
  document.getElementById('download-btn').addEventListener('click', () => {
    if (videoUrl) window.open(videoUrl, '_blank');
  });
}

/* ── Generate ─────────────────────────────────────────────────────────────── */
async function handleGenerate() {
  if (isGenerating) return;
  if (!avatarFile && !audioFile) {
    alert('Please upload both an avatar image and an audio file to generate.');
    return;
  }
  if (!avatarFile) { alert('Please upload an avatar image.'); return; }
  if (!audioFile) { alert('Please upload an audio file.'); return; }

  isGenerating = true;
  const btn = document.getElementById('generate-btn');
  btn.disabled = true;
  btn.querySelector('.btn-text').style.display = 'none';
  btn.querySelector('.btn-loading').style.display = 'inline-flex';

  showProgress(0, 'Initializing...');

  const fd = new FormData();
  fd.append('avatar', avatarFile);
  fd.append('audio', audioFile);
  fd.append('qualityMode', document.getElementById('quality-mode').value);

  const enhancements = Array.from(document.querySelectorAll('[name="enh"]:checked')).map(el => el.value);
  if (enhancements.length) fd.append('enhancements', enhancements.join(','));

  const transcript = document.getElementById('script-input').value.trim();
  if (transcript) fd.append('transcript', transcript);

  try {
    const res = await fetch(`${API_BASE}/render-upload`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    const data = await res.json();
    await pollStatus(data.jobId);
  } catch (err) {
    alert(`Error: ${err.message}`);
    hideProgress();
    resetGenerateBtn();
  }
}

async function pollStatus(jobId) {
  const steps = [
    { p:15, s:'Mapping facial features...' },
    { p:35, s:'Processing voice input...' },
    { p:55, s:'Running lip-sync model...' },
    { p:75, s:'Rendering expressions...' },
    { p:90, s:'Encoding final video...' },
  ];
  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    if (stepIdx < steps.length) { showProgress(steps[stepIdx].p, steps[stepIdx].s); stepIdx++; }
  }, 2000);

  for (let i = 0; i < 300; i++) {
    try {
      const r = await fetch(`${API_BASE}/status/${jobId}`);
      const ct = r.headers.get('content-type') || '';
      if (r.ok && ct.includes('video/mp4')) {
        clearInterval(stepTimer);
        const blob = await r.blob();
        videoUrl = URL.createObjectURL(blob);
        showVideo(videoUrl);
        showProgress(100, 'Complete!');
        setTimeout(hideProgress, 800);
        resetGenerateBtn();
        return;
      }
    } catch (_) { /* retry */ }
    await new Promise(r => setTimeout(r, 1500));
  }

  clearInterval(stepTimer);
  alert('Timed out waiting for generation.');
  hideProgress();
  resetGenerateBtn();
}

/* ── Progress UI ──────────────────────────────────────────────────────────── */
function showProgress(pct, text) {
  const overlay = document.getElementById('progress-overlay');
  overlay.style.display = 'flex';
  document.getElementById('video-placeholder').style.display = 'none';
  const offset = 283 - (pct / 100) * 283;
  document.getElementById('progress-circle').style.strokeDashoffset = offset;
  document.getElementById('progress-pct').textContent = `${pct}%`;
  document.getElementById('progress-status').textContent = text;
}

function hideProgress() {
  document.getElementById('progress-overlay').style.display = 'none';
}

function showVideo(url) {
  const player = document.getElementById('video-player');
  player.src = url;
  player.style.display = 'block';
  document.getElementById('video-placeholder').style.display = 'none';
  document.getElementById('download-btn').disabled = false;
  // Scroll to result
  document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function handleReset() {
  videoUrl = null;
  const player = document.getElementById('video-player');
  player.pause(); player.src = ''; player.style.display = 'none';
  document.getElementById('video-placeholder').style.display = 'flex';
  document.getElementById('download-btn').disabled = true;
  hideProgress();
}

function resetGenerateBtn() {
  isGenerating = false;
  const btn = document.getElementById('generate-btn');
  btn.disabled = false;
  btn.querySelector('.btn-text').style.display = 'inline-flex';
  btn.querySelector('.btn-loading').style.display = 'none';
}

/* ── System Status Check ──────────────────────────────────────────────────── */
async function checkSystemStatus() {
  const container = document.getElementById('status-badges');
  try {
    const res = await fetch(`${API_BASE}/avatars`);
    if (!res.ok) throw new Error('API unreachable');
    const data = await res.json();

    let badges = '';
    const sys = data.system || {};
    if (sys.cuda_available) {
      const gpu = (sys.gpus && sys.gpus[0]) || {};
      badges += `<span class="sys-badge ok">GPU: ${gpu.name || 'Available'} (${gpu.memory_total_gb || '?'} GB)</span>`;
    } else {
      badges += '<span class="sys-badge warn">CPU Mode (no GPU)</span>';
    }

    const models = data.models || {};
    for (const [key, info] of Object.entries(models)) {
      if (info.available) badges += `<span class="sys-badge ok">${info.name || key}</span>`;
      else badges += `<span class="sys-badge err">${info.name || key}</span>`;
    }
    container.innerHTML = badges || '<span class="sys-badge ok">System Ready</span>';
  } catch (e) {
    container.innerHTML = '<span class="sys-badge warn">API Loading...</span>';
    setTimeout(checkSystemStatus, 5000);
  }
}

/* ── Fade-in Observer ─────────────────────────────────────────────────────── */
function initFadeIn() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.1 });
  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
}
