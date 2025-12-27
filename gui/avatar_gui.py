#!/usr/bin/env python3
"""
Avatar Renderer GUI - Multilingual Text-to-Avatar Video Generator

Production Ready Version
- Lazy loading of ML libraries (fixes startup hang)
- Robust WSL/X11 support
- Thread-safe GUI updates
"""

import os
import sys
import ctypes
import ctypes.util

# ============================================================================
# CRITICAL: X11 Threading Fix - MUST BE FIRST!
# This MUST happen before ANY GUI library imports (tkinter, etc.)
# ============================================================================
def initialize_x11_threading():
    """
    Initialize X11 threading support BEFORE any GUI imports.

    This fixes the XCB threading error:
    [xcb] Unknown sequence number while appending request

    CRITICAL: Must be called before importing tkinter or any X11-using library.
    """
    # Check if running in WSL or Linux with X11
    is_wsl = False
    is_linux = sys.platform.startswith('linux')

    try:
        if os.path.exists('/proc/version'):
            with open('/proc/version', 'r') as f:
                is_wsl = 'microsoft' in f.read().lower()
    except Exception:
        pass

    # Only initialize X11 threading on Linux/WSL
    if is_linux or is_wsl:
        # Set environment variables for thread safety
        os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive matplotlib backend
        os.environ['QT_X11_NO_MITSHM'] = '1'  # Disable problematic X11 extension

        # Initialize X11 threading FIRST
        try:
            x11_lib = ctypes.util.find_library('X11')
            if x11_lib:
                x11 = ctypes.cdll.LoadLibrary(x11_lib)
                # CRITICAL: XInitThreads() must be called before any X11 operations
                result = x11.XInitThreads()
                if result == 0:
                    print("[WARN] XInitThreads() returned 0 - threading support may not be available")
                else:
                    print("[INFO] X11 threading initialized successfully")
            else:
                print("[WARN] libX11 not found - running without X11 threading support")
        except Exception as e:
            print(f"[ERROR] Failed to initialize X11 threads: {e}")
            print("[INFO] GUI may experience threading issues")

# Initialize X11 threading BEFORE any other imports
initialize_x11_threading()

# NOW it's safe to import GUI libraries
import threading
import time
import json
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Optional

# ============================================================================
# Dependencies
# ============================================================================

# HTTP Client
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Audio Playback
try:
    import simpleaudio as sa
    HAS_SIMPLEAUDIO = True
except ImportError:
    HAS_SIMPLEAUDIO = False

# Setup path for app modules (but don't import heavy ML libs yet!)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TTS_ENDPOINT = "/text-to-audio"
DEFAULT_AVATAR = ROOT_DIR / "tests/assets/alice.png"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {
    "en": "🇬🇧 English", "es": "🇪🇸 Spanish", "it": "🇮🇹 Italian", "fr": "🇫🇷 French",
    "de": "🇩🇪 German", "ru": "🇷🇺 Russian", "pt": "🇵🇹 Portuguese", "pl": "🇵🇱 Polish",
    "nl": "🇳🇱 Dutch", "sv": "🇸🇪 Swedish", "no": "🇳🇴 Norwegian", "da": "🇩🇰 Danish",
    "fi": "🇫🇮 Finnish", "el": "🇬🇷 Greek", "tr": "🇹🇷 Turkish", "ar": "🇸🇦 Arabic",
    "he": "🇮🇱 Hebrew", "hi": "🇮🇳 Hindi", "ja": "🇯🇵 Japanese", "ko": "🇰🇷 Korean",
    "zh": "🇨🇳 Chinese", "ms": "🇲🇾 Malay", "sw": "🇹🇿 Swahili",
}

VOICE_PROFILES = {
    "🎀 Sophia – Friendly (Female)": {"voice": "female", "temperature": 0.7, "cfg_weight": 0.4, "exaggeration": 0.35, "speed": 1.0},
    "🎀 Emma – Calm (Female)": {"voice": "female", "temperature": 0.6, "cfg_weight": 0.5, "exaggeration": 0.3, "speed": 0.95},
    "🎀 Luna – Energetic (Female)": {"voice": "female", "temperature": 0.8, "cfg_weight": 0.3, "exaggeration": 0.45, "speed": 1.05},
    "👔 Marcus – Professional (Male)": {"voice": "male", "temperature": 0.7, "cfg_weight": 0.4, "exaggeration": 0.3, "speed": 1.0},
    "👔 Ethan – Warm (Male)": {"voice": "male", "temperature": 0.65, "cfg_weight": 0.5, "exaggeration": 0.35, "speed": 0.95},
    "⚪ Neutral Voice": {"voice": "neutral", "temperature": 0.7, "cfg_weight": 0.4, "exaggeration": 0.35, "speed": 1.0},
}

SAMPLE_TEXTS = {
    "en": {"Greeting": "Hello! I'm your AI assistant. How can I help you today?", "Professional": "Thank you for contacting support. I'll be happy to assist you."},
    "es": {"Saludo": "¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?", "Profesional": "Gracias por contactar a soporte. Estaré encantado de asistirte."},
    "it": {"Saluto": "Ciao! Sono il tuo assistente AI. Come posso aiutarti oggi?", "Professionale": "Grazie per aver contattato il supporto."},
    "fr": {"Salutation": "Bonjour! Je suis votre assistant IA.", "Professionnel": "Merci d'avoir contacté notre équipe de support."},
}

QUALITY_MODES = {
    "🔄 Auto (Automatic)": "auto",
    "⚡ Real-time (Fast)": "real_time",
    "✨ High Quality (Best)": "high_quality",
}

# -----------------------------------------------------------------------------
# GUI Application
# -----------------------------------------------------------------------------

class AvatarGeneratorGUI(tk.Tk):
    def __init__(self):
        """
        Initialize the Avatar Renderer GUI.

        Thread-safe initialization with proper X11 support.
        """
        try:
            # Initialize parent Tk class
            super().__init__()

            # Set window properties
            self.title("Avatar Renderer - Production GUI")
            self.geometry("900x850")
            self.resizable(True, True)

            # Thread safety: prevent issues with tcl threading
            try:
                self.tk.call('package', 'require', 'Tcl', '8.5')
            except Exception:
                pass  # Tcl package check is optional

            # State Variables
            self.is_generating = False
            self.avatar_image_path: Optional[Path] = DEFAULT_AVATAR
            self.generated_audio_path: Optional[Path] = None
            self.output_video_path: Optional[Path] = None
            self.current_audio_playback = None
            self.pipeline_loaded = False

            # Build UI
            self._build_widgets()

            # Force the window to show correctly on Windows (fixes taskbar-only issue)
            self.after(50, self._force_show_window)

            # Check API after UI loads
            self.after(1000, self._check_api_health)

        except Exception as e:
            print(f"[ERROR] Failed to initialize GUI: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _build_widgets(self):
        # Styles
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))

        # Main Container
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        ttk.Label(main, text="🎬 Avatar Renderer", style="Header.TLabel").pack(anchor="w")
        ttk.Label(main, text=f"API: {API_BASE_URL}", foreground="gray").pack(anchor="w", pady=(0, 10))

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self._build_text_tab(self.notebook)
        self._build_avatar_tab(self.notebook)
        self._build_generate_tab(self.notebook)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main, variable=self.progress_var, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=(10, 5))

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var, font=("Segoe UI", 9)).pack(anchor="w")

    def _build_text_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="1. Text & Voice")

        # Language & Voice
        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(0, 10))

        # Language
        ttk.Label(controls, text="Language:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar()
        lang_values = sorted([v for k, v in SUPPORTED_LANGUAGES.items()])
        self.lang_combo = ttk.Combobox(controls, textvariable=self.language_var, values=lang_values, state="readonly", width=20)
        self.lang_combo.pack(side=tk.LEFT, padx=(5, 15))
        self.lang_combo.set("🇬🇧 English")
        self.lang_combo.bind("<<ComboboxSelected>>", self._update_sample_buttons)

        # Voice
        ttk.Label(controls, text="Voice:").pack(side=tk.LEFT)
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(controls, textvariable=self.voice_var, values=list(VOICE_PROFILES.keys()), state="readonly", width=30)
        self.voice_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.voice_combo.current(0)

        # Sample Buttons
        self.sample_frame = ttk.Frame(frame)
        self.sample_frame.pack(fill=tk.X, pady=(0, 5))
        self._update_sample_buttons()

        # Text Area
        ttk.Label(frame, text="Text to Speak:").pack(anchor="w")
        self.text_box = tk.Text(frame, height=8, font=("Segoe UI", 11), wrap="word")
        self.text_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.text_box.insert("1.0", "Hello! I am ready to create a video.")

        # Audio Preview
        audio_frame = ttk.LabelFrame(frame, text="Audio Preview", padding=10)
        audio_frame.pack(fill=tk.X)

        self.gen_audio_btn = ttk.Button(audio_frame, text="🎵 Generate Audio", command=self._on_generate_audio)
        self.gen_audio_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.play_audio_btn = ttk.Button(audio_frame, text="▶ Play", command=self._on_play_audio, state=tk.DISABLED)
        self.play_audio_btn.pack(side=tk.LEFT)

    def _build_avatar_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="2. Avatar & Settings")

        # Avatar Selection
        av_frame = ttk.LabelFrame(frame, text="Source Image", padding=10)
        av_frame.pack(fill=tk.X, pady=(0, 10))

        self.avatar_path_var = tk.StringVar(value=str(DEFAULT_AVATAR))
        ttk.Entry(av_frame, textvariable=self.avatar_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(av_frame, text="Browse", command=self._browse_avatar).pack(side=tk.LEFT)

        # Quality
        q_frame = ttk.LabelFrame(frame, text="Render Quality", padding=10)
        q_frame.pack(fill=tk.X, pady=(0, 10))

        self.quality_var = tk.StringVar()
        q_combo = ttk.Combobox(q_frame, textvariable=self.quality_var, values=list(QUALITY_MODES.keys()), state="readonly")
        q_combo.pack(fill=tk.X)
        q_combo.current(0)
        ttk.Label(q_frame, text="Note: 'High Quality' takes significantly longer to render.", foreground="gray", font=("Segoe UI", 8)).pack(anchor="w", pady=(5,0))

        # Output
        out_frame = ttk.LabelFrame(frame, text="Output Folder", padding=10)
        out_frame.pack(fill=tk.X)

        self.out_dir_var = tk.StringVar(value=str(OUTPUT_DIR))
        ttk.Entry(out_frame, textvariable=self.out_dir_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(out_frame, text="Browse", command=self._browse_output).pack(side=tk.LEFT)

    def _build_generate_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="3. Generate")

        # Logs
        self.log_text = tk.Text(frame, font=("Courier", 9), state=tk.DISABLED, height=15)
        scroll = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, in_=self.log_text)

        # Buttons
        btn_frame = ttk.Frame(frame, padding=(0, 10))
        btn_frame.pack(fill=tk.X)

        self.generate_btn = ttk.Button(btn_frame, text="🎬 START RENDER", command=self._on_generate)
        self.generate_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(btn_frame, text="📂 Open Output", command=self._open_output_folder).pack(side=tk.LEFT)

    # -------------------------------------------------------------------------
    # Window Display Fixes (Windows compatibility)
    # -------------------------------------------------------------------------

    def _clamp_to_screen(self):
        """Ensure window is positioned within visible screen area."""
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        # If Tk reports tiny defaults, fall back to intended size
        if w < 200: w = 900
        if h < 200: h = 850

        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _force_show_window(self):
        """Force window to appear and come to front (fixes Windows taskbar-only issue)."""
        # Make sure geometry is calculated
        self.update_idletasks()

        # Clamp window to visible screen area
        self._clamp_to_screen()

        # Ensure it's not minimized/withdrawn
        try:
            self.state("normal")
        except Exception:
            pass
        self.deiconify()

        # Bring to front (common Tk/Windows issue)
        self.lift()
        self.focus_force()

        # Windows-only: temporarily set topmost to force z-order update
        if os.name == "nt":
            self.attributes("-topmost", True)
            self.after(150, lambda: self.attributes("-topmost", False))

    # -------------------------------------------------------------------------
    # Logic & Workers
    # -------------------------------------------------------------------------

    def _get_lang_code(self):
        selected = self.language_var.get()
        for k, v in SUPPORTED_LANGUAGES.items():
            if v == selected: return k
        return "en"

    def _update_sample_buttons(self, event=None):
        for widget in self.sample_frame.winfo_children(): widget.destroy()
        lang = self._get_lang_code()
        samples = SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS["en"])

        for name, text in samples.items():
            btn = ttk.Button(self.sample_frame, text=name,
                           command=lambda t=text: self.text_box.replace("1.0", tk.END, t))
            btn.pack(side=tk.LEFT, padx=(0, 5))

    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _check_api_health(self):
        if not HAS_REQUESTS:
            self.status_var.set("❌ requests lib missing")
            return

        def check():
            try:
                requests.get(f"{API_BASE_URL}/health/live", timeout=3)
                self.after(0, lambda: self.status_var.set("✅ API Connected"))
            except:
                self.after(0, lambda: self.status_var.set("❌ API Offline - Check server"))
        threading.Thread(target=check, daemon=True).start()

    # --- Audio Generation ---
    def _on_generate_audio(self):
        text = self.text_box.get("1.0", tk.END).strip()
        if not text: return messagebox.showwarning("Input", "Please enter text.")

        self.gen_audio_btn.config(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.status_var.set("Generating Audio...")

        threading.Thread(target=self._worker_audio, args=(text,), daemon=True).start()

    def _worker_audio(self, text):
        try:
            lang = self._get_lang_code()
            voice_data = VOICE_PROFILES[self.voice_var.get()]

            payload = {
                "text": text, "language": lang,
                "voice": voice_data["voice"], "temperature": voice_data["temperature"],
                "speed": voice_data["speed"], "output_format": "file"
            }

            resp = requests.post(f"{API_BASE_URL}{TTS_ENDPOINT}", json=payload, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            if data["status"] != "success": raise Exception(data.get("error"))

            audio_path = Path(data["audio_path"])
            if not audio_path.exists(): raise Exception("Audio file missing from server")

            # Move to local output
            local_path = OUTPUT_DIR / f"audio_{int(time.time())}.wav"
            local_path.write_bytes(audio_path.read_bytes())

            self.generated_audio_path = local_path
            self.after(0, self._audio_success)

        except Exception as e:
            self.after(0, lambda: self._error_handler(f"Audio failed: {e}"))

    def _audio_success(self):
        self.gen_audio_btn.config(state=tk.NORMAL)
        self.play_audio_btn.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.status_var.set("✅ Audio Ready")
        messagebox.showinfo("Success", "Audio generated successfully!")

    def _on_play_audio(self):
        if not HAS_SIMPLEAUDIO:
            return messagebox.showinfo("Info", f"Audio saved to: {self.generated_audio_path}\n(Install simpleaudio to play in-app)")
        try:
            wave_obj = sa.WaveObject.from_wave_file(str(self.generated_audio_path))
            wave_obj.play()
        except Exception as e:
            messagebox.showerror("Play Error", str(e))

    # --- Video Generation ---
    def _on_generate(self):
        if not self.avatar_image_path or not Path(self.avatar_image_path).exists():
            return messagebox.showerror("Error", "Please select a valid avatar image.")

        # Audio check
        if not self.generated_audio_path or not self.generated_audio_path.exists():
            # Auto-generate audio if missing
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self._log("Audio not found. Generating audio first...")
            self._on_generate_audio()
            # Wait for audio (simple loop check in thread would be better, but for now we rely on user flow)
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.status_var.set("Rendering Video (This may take time)...")

        # Switch to log tab
        self.notebook.select(2)

        threading.Thread(target=self._worker_render, daemon=True).start()

    def _worker_render(self):
        try:
            self.after(0, lambda: self._log("Initializing pipeline... (First run takes ~10s)"))

            # -------------------------------------------------------
            # LAZY IMPORT - This prevents the GUI from hanging at start
            # -------------------------------------------------------
            try:
                from app.pipeline import render_pipeline
            except ImportError as e:
                raise Exception(f"Failed to import pipeline. Check environment: {e}")

            # Prepare paths
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_out = OUTPUT_DIR / f"avatar_{timestamp}.mp4"

            q_mode = QUALITY_MODES[self.quality_var.get()]

            self.after(0, lambda: self._log(f"Starting render: {q_mode}"))
            self.after(0, lambda: self._log(f"Image: {Path(self.avatar_image_path).name}"))
            self.after(0, lambda: self._log("Processing..."))

            # RUN PIPELINE
            result_path = render_pipeline(
                face_image=str(self.avatar_image_path),
                audio=str(self.generated_audio_path),
                out_path=str(video_out),
                quality_mode=q_mode
            )

            self.output_video_path = Path(result_path)
            self.after(0, self._render_success)

        except Exception as e:
            self.after(0, lambda: self._error_handler(f"Render failed: {e}"))

    def _render_success(self):
        self.generate_btn.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.status_var.set("✅ Render Complete")
        self._log("-" * 30)
        self._log(f"SUCCESS: {self.output_video_path}")
        messagebox.showinfo("Render Complete", f"Video saved to:\n{self.output_video_path}")

    def _error_handler(self, msg):
        self.generate_btn.config(state=tk.NORMAL)
        self.gen_audio_btn.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.status_var.set("❌ Error")
        self._log(f"ERROR: {msg}")
        messagebox.showerror("Error", msg)

    # --- File Helpers ---
    def _browse_avatar(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if f:
            self.avatar_path_var.set(f)
            self.avatar_image_path = f

    def _browse_output(self):
        d = filedialog.askdirectory()
        if d: self.out_dir_var.set(d)

    def _open_output_folder(self):
        path = self.out_dir_var.get()
        if os.name == 'nt': os.startfile(path)
        else: subprocess.call(['xdg-open', path])

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    """
    Main entry point for the GUI application.

    Ensures proper initialization order and thread safety.
    """
    try:
        # Verify X11 threading was initialized (already done at module import)
        print("[INFO] Starting Avatar Renderer GUI...")

        # Additional thread safety: ensure we're on the main thread
        if threading.current_thread() is not threading.main_thread():
            print("[ERROR] GUI must be started from the main thread!")
            return 1

        # Set tkinter to be thread-aware
        try:
            # This helps tkinter handle multi-threaded updates better
            tk._support_default_root = True
        except Exception:
            pass

        # Create and run the application
        app = AvatarGeneratorGUI()

        # Ensure the window is visible and properly initialized
        app.update_idletasks()
        app.update()

        print("[INFO] GUI initialized successfully, starting main loop...")

        # Run the main event loop
        app.mainloop()

        print("[INFO] GUI shutdown cleanly")
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] GUI interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] GUI startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
