"""
Multilingual Optimized GUI client for VRSecretary Chatterbox TTS server.

UPDATED version:
- Fixed ALSA/WSL console spam errors (cannot find card '0').
- robust Audio Device handling (falls back to silent mode if no device found).
- Retains generated audio in memory for replay.
"""

import json
import os
import struct
import threading
import queue
import time
import sys
import contextlib
import ctypes
from typing import Optional, List

import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------------------------------------------------------
# ALSA Error Suppression (Fix for WSL/Linux Console Spam)
# -----------------------------------------------------------------------------
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    # This function intentionally does nothing to suppress ALSA errors
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextlib.contextmanager
def no_alsa_error():
    """Context manager to suppress ALSA C-library errors."""
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except OSError:
        # ALSA not present (e.g., Windows native), just yield
        yield
    except Exception:
        yield

# -----------------------------------------------------------------------------
# Audio Imports (Wrapped in suppression)
# -----------------------------------------------------------------------------
HAS_SIMPLEAUDIO = False
HAS_PYAUDIO = False

with no_alsa_error():
    try:
        import simpleaudio as sa
        HAS_SIMPLEAUDIO = True
    except ImportError:
        pass

    try:
        import pyaudio
        HAS_PYAUDIO = True
    except ImportError:
        pass

if not HAS_PYAUDIO and not HAS_SIMPLEAUDIO:
    print("[WARN] No audio libraries found (pyaudio or simpleaudio). GUI will run in silent mode.")

# HTTP with streaming support
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SERVER_URL = os.getenv("CHATTERBOX_URL", "http://localhost:4123")
STREAMING_ENDPOINT = "/v1/audio/speech/stream"
STANDARD_ENDPOINT = "/v1/audio/speech"
REQUEST_TIMEOUT = 60.0

# Supported languages (23 total)
SUPPORTED_LANGUAGES = {
    "en": "🇬🇧 English",
    "es": "🇪🇸 Spanish",
    "it": "🇮🇹 Italian",
    "fr": "🇫🇷 French",
    "de": "🇩🇪 German",
    "ru": "🇷🇺 Russian",
    "pt": "🇵🇹 Portuguese",
    "pl": "🇵🇱 Polish",
    "nl": "🇳🇱 Dutch",
    "sv": "🇸🇪 Swedish",
    "no": "🇳🇴 Norwegian",
    "da": "🇩🇰 Danish",
    "fi": "🇫🇮 Finnish",
    "el": "🇬🇷 Greek",
    "tr": "🇹🇷 Turkish",
    "ar": "🇸🇦 Arabic",
    "he": "🇮🇱 Hebrew",
    "hi": "🇮🇳 Hindi",
    "ja": "🇯🇵 Japanese",
    "ko": "🇰🇷 Korean",
    "zh": "🇨🇳 Chinese",
    "ms": "🇲🇾 Malay",
    "sw": "🇹🇿 Swahili",
}

# Voice profiles with proper gender configuration
VOICE_PROFILES = {
    "🎀 Sophia – Friendly Assistant (Female)": {
        "voice": "female",
        "temperature": 0.7,
        "cfg_weight": 0.4,
        "exaggeration": 0.35,
        "speed": 1.0,
        "description": "Warm and professional female voice",
    },
    "🎀 Emma – Calm Professional (Female)": {
        "voice": "female",
        "temperature": 0.6,
        "cfg_weight": 0.5,
        "exaggeration": 0.3,
        "speed": 0.95,
        "description": "Composed and clear female voice",
    },
    "🎀 Luna – Energetic & Warm (Female)": {
        "voice": "female",
        "temperature": 0.8,
        "cfg_weight": 0.3,
        "exaggeration": 0.45,
        "speed": 1.05,
        "description": "Lively and expressive female voice",
    },
    "🎀 Maya – Soft & Gentle (Female)": {
        "voice": "female",
        "temperature": 0.65,
        "cfg_weight": 0.45,
        "exaggeration": 0.25,
        "speed": 0.9,
        "description": "Gentle and soothing female voice",
    },
    "👔 Marcus – Professional (Male)": {
        "voice": "male",
        "temperature": 0.7,
        "cfg_weight": 0.4,
        "exaggeration": 0.3,
        "speed": 1.0,
        "description": "Clear and authoritative male voice",
    },
    "👔 Ethan – Warm Baritone (Male)": {
        "voice": "male",
        "temperature": 0.65,
        "cfg_weight": 0.5,
        "exaggeration": 0.35,
        "speed": 0.95,
        "description": "Deep and reassuring male voice",
    },
    "👔 Noah – Neutral & Clear (Male)": {
        "voice": "male",
        "temperature": 0.7,
        "cfg_weight": 0.45,
        "exaggeration": 0.3,
        "speed": 1.0,
        "description": "Balanced and natural male voice",
    },
    "👔 Alex – Dynamic (Male)": {
        "voice": "male",
        "temperature": 0.75,
        "cfg_weight": 0.35,
        "exaggeration": 0.4,
        "speed": 1.05,
        "description": "Energetic and engaging male voice",
    },
    "⚪ Neutral Voice (No Cloning)": {
        "voice": "neutral",
        "temperature": 0.7,
        "cfg_weight": 0.4,
        "exaggeration": 0.35,
        "speed": 1.0,
        "description": "Default synthesized voice",
    },
}

# Sample texts for different languages
SAMPLE_TEXTS = {
    "en": {
        "Greeting": "Hello! I'm your AI assistant. How can I help you today?",
        "Long Text": "The quick brown fox jumps over the lazy dog. This is a test of the text-to-speech system with a longer sentence to demonstrate streaming capabilities and natural voice quality.",
        "Question": "What would you like me to help you with? I'm here to assist with any questions you might have.",
        "Excited": "Wow! This is amazing! The voice sounds so natural and clear!",
        "Professional": "Thank you for contacting our support team. I'll be happy to assist you with your inquiry today.",
    },
    "es": {
        "Saludo": "¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?",
        "Texto Largo": "El veloz murciélago hindú comía feliz cardillo y kiwi. Esta es una prueba del sistema de texto a voz con una oración más larga para demostrar las capacidades de transmisión y la calidad de voz natural.",
        "Pregunta": "¿En qué te gustaría que te ayudara? Estoy aquí para asistirte con cualquier pregunta que tengas.",
        "Emocionado": "¡Guau! ¡Esto es increíble! ¡La voz suena tan natural y clara!",
        "Profesional": "Gracias por contactar a nuestro equipo de soporte. Estaré encantado de asistirte con tu consulta hoy.",
    },
    "it": {
        "Saluto": "Ciao! Sono il tuo assistente AI. Come posso aiutarti oggi?",
        "Testo Lungo": "La volpe veloce marrone salta sopra il cane pigro. Questo è un test del sistema di sintesi vocale con una frase più lunga per dimostrare le capacità di streaming e la qualità vocale naturale.",
        "Domanda": "In cosa vorresti che ti aiutassi? Sono qui per assisterti con qualsiasi domanda tu possa avere.",
        "Emozionato": "Wow! Questo è fantastico! La voce suona così naturale e chiara!",
        "Professionale": "Grazie per aver contattato il nostro team di supporto. Sarò felice di assisterti con la tua richiesta oggi.",
        "Letteratura": "Lascia che te lo dica oggi quanto ti voglio bene quanto tu sei stato sempre per me, come hai arricchito la mia vita. Per te non avrà molta importanza. Tu sei abituato all’amore, esso non è nulla di strano per te sei stato amato e viziato da tante donne. Per me è un’altra cosa. La mia vita è stata povera d’amore, mi è mancato il meglio. Se tuttavia so che cos’è l’amore, è per merito tuo. Te ho potuto amare, te solo fra gli uomini. Tu non puoi misurare ciò che significhi. Significa la sorgente in un deserto l’albero fiorito in un terreno selvaggio. A te solo debbo che il mio cuore non sia inaridito, che sia rimasto in me un punto accessibile alla grazia",
    },
    "fr": {
        "Salutation": "Bonjour! Je suis votre assistant IA. Comment puis-je vous aider aujourd'hui?",
        "Texte Long": "Le renard brun rapide saute par-dessus le chien paresseux. Ceci est un test du système de synthèse vocale avec une phrase plus longue pour démontrer les capacités de streaming et la qualité vocale naturelle.",
        "Question": "En quoi puis-je vous aider? Je suis là pour vous assister avec toutes vos questions.",
        "Excité": "Wow! C'est incroyable! La voix sonne si naturelle et claire!",
        "Professionnel": "Merci d'avoir contacté notre équipe de support. Je serai heureux de vous aider avec votre demande aujourd'hui.",
    },
    "de": {
        "Begrüßung": "Hallo! Ich bin Ihr KI-Assistent. Wie kann ich Ihnen heute helfen?",
        "Langer Text": "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein Test des Text-zu-Sprache-Systems mit einem längeren Satz, um Streaming-Fähigkeiten und natürliche Sprachqualität zu demonstrieren.",
        "Frage": "Wobei möchten Sie, dass ich Ihnen helfe? Ich bin hier, um Sie bei allen Fragen zu unterstützen.",
        "Aufgeregt": "Wow! Das ist erstaunlich! Die Stimme klingt so natürlich und klar!",
        "Professionell": "Vielen Dank, dass Sie unser Support-Team kontaktiert haben. Ich helfe Ihnen gerne bei Ihrer Anfrage heute.",
    },
    "ru": {
        "Приветствие": "Здравствуйте! Я ваш AI-ассистент. Чем я могу помочь вам сегодня?",
        "Длинный Текст": "Быстрая коричневая лиса прыгает через ленивую собаку. Это тест системы преобразования текста в речь с более длинным предложением для демонстрации возможностей потоковой передачи и естественного качества голоса.",
        "Вопрос": "Чем бы вы хотели, чтобы я вам помог? Я здесь, чтобы помочь вам с любыми вопросами.",
        "Восторженный": "Вау! Это потрясающе! Голос звучит так естественно и ясно!",
        "Профессиональный": "Спасибо, что обратились в нашу службу поддержки. Я буду рад помочь вам с вашим запросом сегодня.",
    },
}


# -----------------------------------------------------------------------------
# WAV parsing helpers (fixes unknown format: 3)
# -----------------------------------------------------------------------------

def _parse_wav_chunk(chunk_bytes: bytes) -> dict:
    """Parse a minimal WAV header and return audio parameters + raw data."""

    if len(chunk_bytes) < 44:
        raise ValueError("WAV chunk too small to contain a header")

    header = chunk_bytes[:44]
    (
        chunk_id,
        chunk_size,
        fmt,
        subchunk1_id,
        subchunk1_size,
        audio_format,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        subchunk2_id,
        subchunk2_size,
    ) = struct.unpack("<4sI4s4sIHHIIHH4sI", header)

    if chunk_id != b"RIFF" or fmt != b"WAVE":
        raise ValueError("Invalid WAV header (not RIFF/WAVE)")

    data = chunk_bytes[44:]
    if len(data) == 0:
        raise ValueError("WAV chunk has no data")

    return {
        "audio_format": audio_format,
        "channels": num_channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "data": data,
    }


# -----------------------------------------------------------------------------
# Audio Player (supports streaming chunks + WSL Fallback)
# -----------------------------------------------------------------------------

class StreamingAudioPlayer:
    """Plays audio chunks as they arrive (progressive playback)."""

    def __init__(self):
        self.playing = False
        self.stop_flag = False
        self.audio_queue = queue.Queue()
        self.playback_thread: Optional[threading.Thread] = None

    def start_playback(self):
        """Start the playback thread."""
        self.stop()

        self.playing = True
        self.stop_flag = False
        self.audio_queue = queue.Queue()

        # Prioritize PyAudio, fall back to SimpleAudio, then Dummy
        if HAS_PYAUDIO:
            self.playback_thread = threading.Thread(
                target=self._pyaudio_player,
                daemon=True,
            )
        else:
            self.playback_thread = threading.Thread(
                target=self._simpleaudio_player,
                daemon=True,
            )

        self.playback_thread.start()

    def add_chunk(self, wav_bytes: Optional[bytes]):
        """Add an audio chunk to the playback queue."""
        if self.playing:
            self.audio_queue.put(wav_bytes)

    def stop(self):
        """Stop playback cleanly."""
        self.stop_flag = True
        if self.playback_thread and self.playback_thread.is_alive():
            self.audio_queue.put(None)
            try:
                self.playback_thread.join(timeout=2.0)
            except RuntimeError:
                pass
        self.playback_thread = None
        self.playing = False

    def is_alive(self):
        return self.playback_thread and self.playback_thread.is_alive()

    def _pyaudio_player(self):
        """Real-time streaming playback using PyAudio with WSL protection."""
        if not HAS_PYAUDIO:
            return

        p = None
        stream = None
        device_ok = False

        # Attempt to initialize PyAudio inside suppressed ALSA context
        with no_alsa_error():
            try:
                p = pyaudio.PyAudio()
                # Try to open a dummy stream to check if device actually works
                try:
                    dummy = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
                    dummy.close()
                    device_ok = True
                except Exception:
                    print("\n[INFO] No audio output device detected (Silent Mode active).")
                    device_ok = False
            except Exception as e:
                print(f"[WARN] PyAudio initialization failed: {e}")
                device_ok = False

        try:
            while not self.stop_flag:
                chunk_bytes = self.audio_queue.get()
                if chunk_bytes is None:
                    break

                if not device_ok:
                    # FALLBACK: Just consume the queue so the UI updates
                    continue

                try:
                    info = _parse_wav_chunk(chunk_bytes)
                except Exception as exc:
                    print(f"[WARN] Failed to parse WAV chunk: {exc}")
                    continue

                audio_format = info["audio_format"]
                channels = info["channels"]
                sample_rate = info["sample_rate"]
                bits_per_sample = info["bits_per_sample"]
                frames = info["data"]

                if audio_format == 1:  # PCM
                    sample_width = bits_per_sample // 8
                    pa_format = p.get_format_from_width(sample_width)
                elif audio_format == 3:  # IEEE float
                    pa_format = pyaudio.paFloat32
                else:
                    continue

                if stream is None:
                    try:
                        stream = p.open(
                            format=pa_format,
                            channels=channels,
                            rate=sample_rate,
                            output=True,
                        )
                    except OSError:
                        # Stream failed mid-init, switch to silent mode
                        device_ok = False
                        continue

                if stream:
                    stream.write(frames)

        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if p:
                p.terminate()
            self.playing = False

    def _simpleaudio_player(self):
        """Sequential playback using simpleaudio."""
        if not HAS_SIMPLEAUDIO:
            return

        try:
            while not self.stop_flag:
                chunk_bytes = self.audio_queue.get()
                if chunk_bytes is None:
                    break

                try:
                    info = _parse_wav_chunk(chunk_bytes)
                except Exception:
                    continue

                frames = info["data"]
                channels = info["channels"]
                sample_rate = info["sample_rate"]
                bits_per_sample = info["bits_per_sample"]

                try:
                    wave_obj = sa.WaveObject(
                        frames,
                        channels,
                        bits_per_sample // 8,
                        sample_rate,
                    )
                    play_obj = wave_obj.play()
                    
                    while play_obj.is_playing() and not self.stop_flag:
                         time.sleep(0.1)
                    
                    if self.stop_flag and play_obj.is_playing():
                        play_obj.stop()
                        
                except Exception:
                    # Audio device fail, consume queue
                    while not self.audio_queue.empty():
                        self.audio_queue.get()
                    break

        finally:
            self.playing = False


# -----------------------------------------------------------------------------
# TTS Client (with language support)
# -----------------------------------------------------------------------------

def call_tts_streaming(
    text: str,
    language: str,  # Language parameter
    profile: dict,
    chunk_by_sentences: bool = True,
    progress_callback=None,
    cancel_event: Optional[threading.Event] = None,
    response_holder: Optional[dict] = None,
) -> int:
    """Call the streaming TTS endpoint with language support."""

    if not HAS_REQUESTS:
        raise RuntimeError("The 'requests' library is required.")

    url = SERVER_URL.rstrip("/") + STREAMING_ENDPOINT

    payload = {
        "input": text,
        "language": language,
        "voice": profile.get("voice", "neutral"),
        "temperature": profile.get("temperature", 0.7),
        "cfg_weight": profile.get("cfg_weight", 0.4),
        "exaggeration": profile.get("exaggeration", 0.35),
        "speed": profile.get("speed", 1.0),
        "stream": True,
        "chunk_by_sentences": bool(chunk_by_sentences),
    }

    response = None
    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=10, # Connect timeout
        )
        response.raise_for_status()

        if response_holder is not None:
            response_holder["response"] = response

        chunk_num = 0
        for chunk in response.iter_content(chunk_size=None):
            if cancel_event is not None and cancel_event.is_set():
                break

            if not chunk:
                continue
            chunk_num += 1
            if progress_callback:
                progress_callback(chunk_num, chunk)

        return chunk_num

    except requests.exceptions.RequestException as e:
        if cancel_event is not None and cancel_event.is_set():
            return 0
        raise RuntimeError(f"Streaming request failed: {e}") from e

    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if response_holder is not None:
            response_holder["response"] = None


def call_tts_standard(
    text: str,
    language: str,  # Language parameter
    profile: dict,
    chunk_by_sentences: bool = False,
) -> bytes:
    """Call the standard (non-streaming) TTS endpoint with language support."""

    url = SERVER_URL.rstrip("/") + STANDARD_ENDPOINT

    payload = {
        "input": text,
        "language": language,
        "voice": profile.get("voice", "neutral"),
        "temperature": profile.get("temperature", 0.7),
        "cfg_weight": profile.get("cfg_weight", 0.4),
        "exaggeration": profile.get("exaggeration", 0.35),
        "speed": profile.get("speed", 1.0),
        "stream": False,  # non-streaming mode by default
        "chunk_by_sentences": bool(chunk_by_sentences),
    }

    if HAS_REQUESTS:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.content
    else:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read()


# -----------------------------------------------------------------------------
# GUI Application (with language selector)
# -----------------------------------------------------------------------------

class MultilingualVoiceTestApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("VRSecretary TTS - Multilingual Voice Test")
        self.geometry("800x650")
        self.resizable(True, True)

        self.player = StreamingAudioPlayer()
        self.is_generating = False
        self.is_replaying = False

        self.last_audio_cache: List[bytes] = []
        self.streaming_enabled = tk.BooleanVar(value=False)
        self.chunking_enabled = tk.BooleanVar(value=True)

        self.current_cancel_event: Optional[threading.Event] = None
        self.current_response_holder: dict = {}

        self._build_widgets()
        self._check_server_health()

    # ----------------- UI building -----------------

    def _build_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            main,
            text="🌍 VRSecretary TTS - Multilingual Voice Test",
            font=("Segoe UI", 14, "bold"),
        )
        header.pack(anchor="w", pady=(0, 5))

        server_info = ttk.Label(
            main,
            text=f"Server: {SERVER_URL} | 23 Languages Supported",
            foreground="gray",
        )
        server_info.pack(anchor="w")

        # Language selection
        lang_frame = ttk.LabelFrame(main, text="Language Selection", padding=10)
        lang_frame.pack(fill=tk.X, pady=(10, 0))

        lang_inner = ttk.Frame(lang_frame)
        lang_inner.pack(fill=tk.X)

        ttk.Label(lang_inner, text="Language:").pack(side=tk.LEFT)

        self.language_var = tk.StringVar()
        lang_items = sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])
        lang_display_names = [name for _, name in lang_items]
        self.lang_codes = [code for code, _ in lang_items]

        self.language_combo = ttk.Combobox(
            lang_inner,
            textvariable=self.language_var,
            values=lang_display_names,
            state="readonly",
            width=25,
        )
        self.language_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.language_combo.set("🇬🇧 English")
        self.language_combo.bind("<<ComboboxSelected>>", self._on_language_changed)

        lang_info = ttk.Label(
            lang_frame,
            text="ℹ Select the language for speech synthesis",
            foreground="gray",
            font=("Segoe UI", 8, "italic"),
        )
        lang_info.pack(anchor="w", pady=(5, 0))

        # Voice selection
        voice_frame = ttk.LabelFrame(main, text="Voice Selection", padding=10)
        voice_frame.pack(fill=tk.X, pady=(10, 0))

        voice_inner = ttk.Frame(voice_frame)
        voice_inner.pack(fill=tk.X)

        ttk.Label(voice_inner, text="Voice Profile:").pack(side=tk.LEFT)

        self.voice_var = tk.StringVar()
        voice_names = list(VOICE_PROFILES.keys())

        self.voice_combo = ttk.Combobox(
            voice_inner,
            textvariable=self.voice_var,
            values=voice_names,
            state="readonly",
            width=40,
        )
        self.voice_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.voice_combo.set(voice_names[0])
        self.voice_combo.bind("<<ComboboxSelected>>", self._on_voice_changed)

        self.voice_desc_var = tk.StringVar()
        self.voice_desc_label = ttk.Label(
            voice_frame,
            textvariable=self.voice_desc_var,
            foreground="gray",
            font=("Segoe UI", 9, "italic"),
        )
        self.voice_desc_label.pack(anchor="w", pady=(5, 0))
        self._on_voice_changed()

        # Sample texts
        sample_frame = ttk.Frame(main)
        sample_frame.pack(fill=tk.X, pady=(10, 5))

        ttk.Label(sample_frame, text="Quick samples:").pack(side=tk.LEFT)

        self.sample_buttons_frame = ttk.Frame(sample_frame)
        self.sample_buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._update_sample_buttons()

        # Text input
        text_frame = ttk.LabelFrame(main, text="Text to Synthesize", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        text_container = ttk.Frame(text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)

        self.text_box = tk.Text(
            text_container,
            wrap="word",
            font=("Segoe UI", 10),
            height=8,
        )
        scrollbar = ttk.Scrollbar(text_container, command=self.text_box.yview)
        self.text_box.config(yscrollcommand=scrollbar.set)

        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        default_text = SAMPLE_TEXTS.get("en", {}).get("Greeting", "Hello!")
        self.text_box.insert("1.0", default_text)

        # Controls
        chunk_frame = ttk.Frame(main)
        chunk_frame.pack(fill=tk.X, pady=(5, 0))

        self.streaming_check = ttk.Checkbutton(
            chunk_frame,
            text="Use streaming API (progressive playback)",
            variable=self.streaming_enabled,
            command=self._on_streaming_toggle,
        )
        self.streaming_check.pack(side=tk.LEFT)

        self.chunk_check = ttk.Checkbutton(
            chunk_frame,
            text="Enable sentence-based chunking",
            variable=self.chunking_enabled,
        )
        self.chunk_check.pack(side=tk.LEFT, padx=(10, 0))

        chunk_hint = ttk.Label(
            chunk_frame,
            text="Chunking applies only when streaming is enabled",
            foreground="gray",
            font=("Segoe UI", 8, "italic"),
        )
        chunk_hint.pack(side=tk.LEFT, padx=(5, 0))
        self._on_streaming_toggle()

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main,
            variable=self.progress_var,
            mode="indeterminate",
        )
        self.progress_bar.pack(fill=tk.X, pady=(10, 5))

        bottom_frame = ttk.Frame(main)
        bottom_frame.pack(fill=tk.X, pady=(5, 0))

        status_frame = ttk.Frame(bottom_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side=tk.LEFT)

        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(side=tk.RIGHT)

        self.stop_button = ttk.Button(
            button_frame,
            text="⏹ Stop",
            command=self._on_stop,
            state=tk.DISABLED,
            width=10,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))

        self.replay_button = ttk.Button(
            button_frame,
            text="♻ Play Again",
            command=self._on_replay,
            state=tk.DISABLED,
            width=12,
        )
        self.replay_button.pack(side=tk.LEFT, padx=(0, 5))

        self.speak_button = ttk.Button(
            button_frame,
            text="🎙 Speak",
            command=self._on_speak,
            width=12,
        )
        self.speak_button.pack(side=tk.LEFT)

    # ----------------- Logic -----------------

    def _get_current_language_code(self) -> str:
        display_name = self.language_var.get()
        for code, name in SUPPORTED_LANGUAGES.items():
            if name == display_name:
                return code
        return "en"

    def _on_language_changed(self, event=None):
        self._update_sample_buttons()

    def _update_sample_buttons(self):
        for widget in self.sample_buttons_frame.winfo_children():
            widget.destroy()

        lang_code = self._get_current_language_code()
        samples = SAMPLE_TEXTS.get(lang_code, SAMPLE_TEXTS.get("en", {}))

        for sample_name in samples.keys():
            btn = ttk.Button(
                self.sample_buttons_frame,
                text=sample_name,
                command=lambda name=sample_name, lang=lang_code: self._load_sample(name, lang),
                width=12,
            )
            btn.pack(side=tk.LEFT, padx=(5, 0))

    def _load_sample(self, sample_name: str, lang_code: str):
        samples = SAMPLE_TEXTS.get(lang_code, SAMPLE_TEXTS.get("en", {}))
        text = samples.get(sample_name, "")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)

    def _on_streaming_toggle(self):
        streaming = bool(self.streaming_enabled.get())
        if streaming:
            self.chunk_check.configure(state=tk.NORMAL)
        else:
            self.chunk_check.configure(state=tk.DISABLED)

    def _on_voice_changed(self, event=None):
        voice_name = self.voice_var.get()
        profile = VOICE_PROFILES.get(voice_name, {})
        desc = profile.get("description", "No description available")
        self.voice_desc_var.set(f"ℹ {desc}")

    def _check_server_health(self):
        def check():
            try:
                health_url = SERVER_URL.rstrip("/") + "/health"
                if HAS_REQUESTS:
                    response = requests.get(health_url, timeout=5)
                    data = response.json()
                else:
                    import urllib.request
                    with urllib.request.urlopen(health_url, timeout=5) as resp:
                        data = json.loads(resp.read())

                status = data.get("status", "unknown")
                device = data.get("device", "unknown")
                supported_langs = data.get("supported_languages", {})
                lang_count = len(supported_langs) if isinstance(supported_langs, dict) else 23

                msg = f"✓ Server connected ({device}, {lang_count} languages)"
                if status != "ready":
                    msg = "⚠ Server initializing..."

                self.after(0, lambda: self.status_var.set(msg))

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self.status_var.set(f"❌ Server offline: {err_msg}"))

        threading.Thread(target=check, daemon=True).start()

    def _cancel_current_generation(self):
        if self.current_cancel_event is not None:
            self.current_cancel_event.set()

        resp = self.current_response_holder.get("response") if self.current_response_holder else None
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

        self.player.stop()

    def _on_speak(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to synthesize.")
            return

        voice_name = self.voice_var.get()
        profile = VOICE_PROFILES.get(voice_name, {})
        language = self._get_current_language_code()

        if self.is_generating or self.is_replaying:
            self._cancel_current_generation()

        self.is_generating = True
        self.is_replaying = False
        
        self.last_audio_cache = []

        self.speak_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar.start(10)

        use_streaming = bool(self.streaming_enabled.get())
        chunk_by_sentences = bool(self.chunking_enabled.get()) if use_streaming else False

        self.player.start_playback()

        cancel_event = threading.Event()
        self.current_cancel_event = cancel_event
        self.current_response_holder = {}

        threading.Thread(
            target=self._worker_speak,
            args=(
                text,
                language,
                profile,
                voice_name,
                use_streaming,
                chunk_by_sentences,
                cancel_event,
                self.current_response_holder,
            ),
            daemon=True,
        ).start()

    def _worker_speak(
        self,
        text: str,
        language: str,
        profile: dict,
        voice_name: str,
        use_streaming: bool,
        chunk_by_sentences: bool,
        cancel_event: threading.Event,
        response_holder: dict,
    ):
        try:
            lang_display = SUPPORTED_LANGUAGES.get(language, language)
            accumulated_chunks = []

            if use_streaming:
                mode_label = "streaming, sentence-chunked" if chunk_by_sentences else "streaming, full-text"
                self.after(0, lambda: self.status_var.set(f"🎵 Generating {voice_name} ({lang_display}, {mode_label})..."))

                def on_chunk(chunk_num, chunk_bytes):
                    if cancel_event.is_set():
                        return
                    self.player.add_chunk(chunk_bytes)
                    accumulated_chunks.append(chunk_bytes)
                    self.after(0, lambda: self.status_var.set(f"🎵 Playing chunk {chunk_num} ({lang_display}, {mode_label})..."))

                total_chunks = call_tts_streaming(
                    text,
                    language,
                    profile,
                    chunk_by_sentences=chunk_by_sentences,
                    progress_callback=on_chunk,
                    cancel_event=cancel_event,
                    response_holder=response_holder,
                )

                if cancel_event.is_set():
                    return

                self.player.add_chunk(None)
                self.last_audio_cache = accumulated_chunks

                self.after(0, self._on_speak_complete, total_chunks, voice_name, language, True, chunk_by_sentences)
            else:
                mode_label = "non-streaming (full file)"
                self.after(0, lambda: self.status_var.set(f"🎵 Generating {voice_name} ({lang_display}, {mode_label})..."))

                wav_bytes = call_tts_standard(
                    text,
                    language,
                    profile,
                    chunk_by_sentences=False,
                )

                if cancel_event.is_set():
                    return

                self.player.add_chunk(wav_bytes)
                self.player.add_chunk(None)
                self.last_audio_cache = [wav_bytes]

                self.after(0, self._on_speak_complete, 1, voice_name, language, False, False)

        except Exception as e:
            if cancel_event.is_set():
                return
            self.after(0, self._on_speak_error, str(e))

    def _on_speak_complete(self, chunk_count, voice_name, language, use_streaming, chunk_by_sentences):
        self.is_generating = False
        self.is_replaying = False
        self.current_cancel_event = None
        self.current_response_holder = {}

        self.speak_button.config(state=tk.NORMAL)
        if self.last_audio_cache:
            self.replay_button.config(state=tk.NORMAL)
            
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()

        if use_streaming:
            mode_label = "streaming, sentence-chunked" if chunk_by_sentences else "streaming, full-text"
        else:
            mode_label = "non-streaming (full file)"

        lang_display = SUPPORTED_LANGUAGES.get(language, language)
        self.status_var.set(f"✓ Completed ({chunk_count} chunk(s), {voice_name}, {lang_display}, {mode_label})")

    def _on_speak_error(self, error_msg: str):
        self.is_generating = False
        self.is_replaying = False
        self.current_cancel_event = None
        self.current_response_holder = {}

        self.speak_button.config(state=tk.NORMAL)
        self.replay_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.player.stop()
        self.status_var.set("❌ Error occurred")
        messagebox.showerror("TTS Error", f"Failed to generate speech:\n\n{error_msg}")

    def _on_replay(self):
        if not self.last_audio_cache:
            return
            
        if self.is_generating or self.is_replaying:
            self.player.stop()
            
        self.is_replaying = True
        self.status_var.set("♻ Replaying cached audio...")
        
        self.speak_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.player.start_playback()
        for chunk in self.last_audio_cache:
            self.player.add_chunk(chunk)
        self.player.add_chunk(None)
        
        def watcher():
            while self.player.is_alive():
                time.sleep(0.1)
            self.after(0, self._on_replay_complete)
            
        threading.Thread(target=watcher, daemon=True).start()

    def _on_replay_complete(self):
        self.is_replaying = False
        self.speak_button.config(state=tk.NORMAL)
        self.replay_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("✓ Replay finished")

    def _on_stop(self):
        if not self.is_generating and not self.is_replaying:
            return

        self._cancel_current_generation()
        
        self.is_generating = False
        self.is_replaying = False
        self.current_cancel_event = None
        self.current_response_holder = {}

        self.speak_button.config(state=tk.NORMAL)
        if self.last_audio_cache:
             self.replay_button.config(state=tk.NORMAL)
        else:
             self.replay_button.config(state=tk.DISABLED)
             
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.status_var.set("⏹ Stopped")


def main():
    missing = []
    if not HAS_REQUESTS:
        missing.append("requests")
    if not HAS_SIMPLEAUDIO and not HAS_PYAUDIO:
        missing.append("simpleaudio or pyaudio")

    if missing:
        print("\n⚠ WARNING: Missing optional dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        print("\nThe app will work with limited functionality.\n")

    app = MultilingualVoiceTestApp()
    app.mainloop()


if __name__ == "__main__":
    main()