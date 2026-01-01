# app/tts/chatterbox_server.py
from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, AsyncIterator, Literal, Optional

# -----------------------------------------------------------------------------
# WARNING FILTERS (Clean up logs)
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", message=".*return_dict_in_generate.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# IMPORTANT: Import the multilingual version
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    print("CRITICAL ERROR: Could not import 'chatterbox.mtl_tts'. Ensure your python path is correct.")
    raise

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("vr_chatterbox_server_multilingual")

SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(BASE_DIR, "voices")

# Performance tuning
MAX_WORKERS = int(os.getenv("CHATTERBOX_MAX_WORKERS", "4"))
PCM_STREAM = os.getenv("CHATTERBOX_STREAM_PCM", "true").lower() == "true"

# Long-form chunking defaults (blog posts / pages)
DEFAULT_CHUNK_WORDS = int(os.getenv("CHATTERBOX_CHUNK_WORDS", "90"))
DEFAULT_MAX_SENTENCES = int(os.getenv("CHATTERBOX_MAX_SENTENCES", "10"))
ADD_GAP_MS = int(os.getenv("CHATTERBOX_GAP_MS", "120"))  # silence between chunks in non-stream WAV

# Voice fine-tuning
FEMALE_EXAGGERATION_BASE = float(os.getenv("CHATTERBOX_FEMALE_EXAGGERATION", "0.45"))
MALE_EXAGGERATION_BASE = float(os.getenv("CHATTERBOX_MALE_EXAGGERATION", "0.30"))

# Post-processing (standard production fixes)
TRIM_END_SILENCE = os.getenv("CHATTERBOX_TRIM_END_SILENCE", "true").lower() == "true"
TRIM_DB = float(os.getenv("CHATTERBOX_TRIM_DB", "-45"))
TRIM_PAD_MS = int(os.getenv("CHATTERBOX_TRIM_PAD_MS", "80"))
FADE_MS = int(os.getenv("CHATTERBOX_FADE_MS", "12"))
HIGHPASS_HZ = int(os.getenv("CHATTERBOX_HIGHPASS_HZ", "25"))

# Generation guardrails / retries (prevents "one-word" outputs)
MIN_AUDIO_SEC_FLOOR = float(os.getenv("CHATTERBOX_MIN_AUDIO_SEC_FLOOR", "0.45"))
MIN_AUDIO_FRAC_OF_EXPECTED = float(os.getenv("CHATTERBOX_MIN_AUDIO_FRAC_OF_EXPECTED", "0.35"))
MAX_RETRIES_SHORT_AUDIO = int(os.getenv("CHATTERBOX_MAX_RETRIES_SHORT_AUDIO", "2"))

# Disable Chatterbox's AlignmentStreamAnalyzer (it can force EOS early -> "one word")
DISABLE_ALIGNMENT_ANALYZER = os.getenv("CHATTERBOX_DISABLE_ALIGNMENT_ANALYZER", "true").lower() == "true"

# Transformers attention implementation (removes SDPA+output_attentions warning)
ATTN_IMPL = os.getenv("CHATTERBOX_ATTN_IMPL", "eager")  # eager | sdpa | flash_attention_2 (if supported)

# Sentence splitting
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[\.!?。！？])\s+")
PARA_SPLIT_PATTERN = re.compile(r"\n\s*\n+")


def configure_logging(level: str = "INFO") -> None:
    level = level.upper()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=getattr(logging, level, logging.INFO),
    )
    logger.setLevel(getattr(logging, level, logging.INFO))


def detect_device(explicit: Optional[str] = None) -> str:
    """Auto-detect best available device."""
    if explicit:
        return explicit
    env_dev = os.getenv("CHATTERBOX_DEVICE")
    if env_dev:
        return env_dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
VoiceType = Literal["female", "male", "neutral"]
LanguageCode = Literal[
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
]


class SpeechRequest(BaseModel):
    """Request schema for multilingual TTS generation."""
    input: str = Field(..., description="Text to synthesize.", min_length=1, max_length=200_000)
    language: LanguageCode = Field("en", description="Language code for synthesis.")
    voice: VoiceType = Field("neutral", description="Voice profile: 'female', 'male', or 'neutral'.")

    # Generation parameters
    temperature: float = Field(0.8, ge=0.05, le=2.0, description="Sampling temperature.")
    cfg_weight: float = Field(0.4, ge=0.0, le=2.0, description="Guidance weight.")
    exaggeration: float = Field(0.3, ge=0.0, le=1.0, description="Expressiveness level.")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speed multiplier (resample: changes pitch).")

    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    top_k: int = Field(50, ge=0, description="Top-K sampling.")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Penalty for repetition.")

    # Critical for long-form stability / preventing early stop
    max_new_tokens: int = Field(1200, ge=64, le=6000, description="Max speech tokens to generate.")
    min_new_tokens: int = Field(0, ge=0, le=2000, description="Min speech tokens to generate (if supported).")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="Optional RNG seed for retries.")

    # Streaming options
    stream: bool = Field(True, description="Enable streaming.")
    chunk_by_sentences: bool = Field(True, description="Chunk long text (recommended for long-form).")
    chunk_words: Optional[int] = Field(None, ge=10, le=2000, description="Target words per chunk.")
    max_chunk_sentences: Optional[int] = Field(None, ge=1, le=200, description="Max sentences per chunk.")
    preserve_paragraphs: bool = Field(True, description="Prefer chunk boundaries at blank lines.")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    model_ready: bool
    voices_loaded: dict[str, bool]
    active_requests: int
    supported_languages: dict[str, str]


class LanguagesResponse(BaseModel):
    languages: dict[str, str]
    count: int


# -----------------------------------------------------------------------------
# Voice Profile Manager (Multilingual)
# -----------------------------------------------------------------------------
class VoiceProfileManager:
    """Manages pre-loaded voice profiles with language awareness."""

    def __init__(self, model: ChatterboxMultilingualTTS):
        self.model = model

        if not os.path.exists(VOICES_DIR):
            try:
                os.makedirs(VOICES_DIR, exist_ok=True)
            except OSError:
                pass

        default_female = os.path.join(VOICES_DIR, "female.wav")
        default_male = os.path.join(VOICES_DIR, "male.wav")

        self.profiles: dict[str, Optional[str]] = {
            "female": os.getenv("CHATTERBOX_FEMALE_VOICE", default_female),
            "male": os.getenv("CHATTERBOX_MALE_VOICE", default_male),
        }
        self.prepared: dict[str, bool] = {}
        self._lock = threading.Lock()

    def load_all(self) -> None:
        """Pre-load all available voice profiles."""
        for voice_type, path in self.profiles.items():
            if path:
                path = os.path.abspath(path)
                self.profiles[voice_type] = path

            if path and os.path.exists(path):
                try:
                    logger.info("Loading %s voice profile from: %s", voice_type, path)
                    prep_exaggeration = (
                        FEMALE_EXAGGERATION_BASE if voice_type == "female" else MALE_EXAGGERATION_BASE
                    )
                    self.model.prepare_conditionals(path, exaggeration=prep_exaggeration)
                    self.prepared[voice_type] = True
                    logger.info("✓ %s voice profile loaded", voice_type.capitalize())
                except Exception as exc:
                    logger.warning("Failed to load %s voice: %s", voice_type, exc)
                    self.prepared[voice_type] = False
            else:
                self.prepared[voice_type] = False
                if path:
                    logger.warning(
                        "%s voice file not found at: %s (Will fallback to neutral)",
                        voice_type.capitalize(),
                        path,
                    )

    def get_voice_path(self, voice: VoiceType) -> Optional[str]:
        """Get the path for a specific voice type. Fallback to None (neutral) if missing."""
        if voice == "neutral":
            return None
        path = self.profiles.get(voice)
        if path and os.path.exists(path):
            return path
        return None

    def is_loaded(self, voice: VoiceType) -> bool:
        if voice == "neutral":
            return True
        return self.prepared.get(voice, False)


# -----------------------------------------------------------------------------
# Multilingual TTS Service
# -----------------------------------------------------------------------------
class ChatterboxServiceMultilingual:
    """Production-ready multilingual TTS service with long-form chunking and anti-truncation retries."""

    def __init__(self, device: str):
        self.device = device
        self._model: Optional[ChatterboxMultilingualTTS] = None
        self._voice_manager: Optional[VoiceProfileManager] = None
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="tts-worker")
        self._active_requests = 0
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._initialized = False

        self._generate_sig_params: Optional[set[str]] = None

    @staticmethod
    def _patch_alignment_stream_analyzer() -> None:
        """
        Prevent "one-word" outputs caused by AlignmentStreamAnalyzer forcing EOS when it detects repetition.
        Robust patch across chatterbox versions by accepting *args/**kwargs.
        """
        if not DISABLE_ALIGNMENT_ANALYZER:
            return
        try:
            from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

            def _add_attention_spy_noop(self, *args, **kwargs):  # noqa: ANN001
                return None

            def _step_noop(self, *args, **kwargs):  # noqa: ANN001
                # Some versions call step(logits), others may pass extra args.
                # Return logits unchanged (first positional arg), or if absent return None.
                return args[0] if args else None

            # Patch both known method names (defensive)
            if hasattr(AlignmentStreamAnalyzer, "_add_attention_spy"):
                AlignmentStreamAnalyzer._add_attention_spy = _add_attention_spy_noop  # type: ignore[attr-defined]
            if hasattr(AlignmentStreamAnalyzer, "_add_attention_spy_noop"):
                AlignmentStreamAnalyzer._add_attention_spy_noop = _add_attention_spy_noop  # type: ignore[attr-defined]
            if hasattr(AlignmentStreamAnalyzer, "step"):
                AlignmentStreamAnalyzer.step = _step_noop  # type: ignore[assignment]

            logger.warning("🩹 AlignmentStreamAnalyzer disabled (prevents forced-EOS truncation).")
        except Exception as e:
            logger.warning("Could not patch AlignmentStreamAnalyzer: %s", e)

    @staticmethod
    def _from_pretrained_kwargs() -> dict[str, Any]:
        """
        Pass only supported kwargs to ChatterboxMultilingualTTS.from_pretrained.
        This removes the SDPA output_attentions warning when 'attn_implementation' is supported.
        """
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(ChatterboxMultilingualTTS.from_pretrained)
            params = set(sig.parameters.keys())
            if "attn_implementation" in params and ATTN_IMPL:
                kwargs["attn_implementation"] = ATTN_IMPL
        except Exception:
            pass
        return kwargs

    def initialize(self) -> None:
        """Initialize multilingual model and voice profiles."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            logger.info("🚀 Initializing ChatterboxMultilingualTTS on %s...", self.device)
            start = time.time()

            try:
                # Patch BEFORE model instantiation
                self._patch_alignment_stream_analyzer()

                fp_kwargs = self._from_pretrained_kwargs()
                if fp_kwargs.get("attn_implementation"):
                    logger.info("🧠 attn_implementation: %s", fp_kwargs["attn_implementation"])

                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, **fp_kwargs)
                if hasattr(self._model, "eval"):
                    self._model.eval()

                torch.set_grad_enabled(False)

                # cache signature of generate() so we only pass supported kwargs
                try:
                    self._generate_sig_params = set(inspect.signature(self._model.generate).parameters.keys())
                except Exception:
                    self._generate_sig_params = None

                self._voice_manager = VoiceProfileManager(self._model)
                self._voice_manager.load_all()

                self._initialized = True
                elapsed = time.time() - start
                logger.info("✓ ChatterboxMultilingualTTS ready in %.2f seconds", elapsed)
                logger.info("📚 Supported languages: %d", len(SUPPORTED_LANGUAGES))
            except Exception as e:
                logger.error("Failed to initialize model: %s", e)
                raise

    def shutdown(self) -> None:
        logger.info("🛑 Shutting down TTS executor...")
        self._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("🛑 TTS executor shutdown complete.")

    @property
    def model(self) -> ChatterboxMultilingualTTS:
        if not self._initialized or self._model is None:
            self.initialize()
        assert self._model is not None
        return self._model

    @property
    def voice_manager(self) -> VoiceProfileManager:
        if not self._initialized or self._voice_manager is None:
            self.initialize()
        assert self._voice_manager is not None
        return self._voice_manager

    @staticmethod
    def _approx_word_count(text: str) -> int:
        basic = len(text.split())
        if basic > 0:
            return basic
        return max(1, len(text) // 4)

    def _split_into_sentences(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        spans: list[tuple[int, int]] = []
        last_idx = 0
        for match in SENTENCE_SPLIT_PATTERN.finditer(text):
            end_idx = match.start()
            if end_idx > last_idx:
                spans.append((last_idx, end_idx))
                last_idx = end_idx
        if last_idx < len(text):
            spans.append((last_idx, len(text)))
        return spans

    def _chunk_text(
        self,
        text: str,
        word_target: int,
        max_sentences: Optional[int],
        preserve_paragraphs: bool,
    ) -> list[str]:
        if preserve_paragraphs:
            paras = [p.strip() for p in PARA_SPLIT_PATTERN.split(text) if p.strip()]
        else:
            paras = [text]

        chunks: list[str] = []
        for para in paras:
            spans = self._split_into_sentences(para)
            if not spans:
                chunks.append(para)
                continue

            current_start: Optional[int] = None
            current_word_count = 0
            current_sentence_count = 0

            def flush(end_index: int) -> None:
                nonlocal current_start, current_word_count, current_sentence_count
                if current_start is not None and end_index > current_start:
                    chunks.append(para[current_start:end_index].strip())
                current_start = None
                current_word_count = 0
                current_sentence_count = 0

            for s_start, s_end in spans:
                sent_text = para[s_start:s_end]
                sent_words = self._approx_word_count(sent_text)

                if current_sentence_count == 0:
                    current_start = s_start

                new_word_count = current_word_count + sent_words
                new_sentence_count = current_sentence_count + 1

                if current_sentence_count > 0 and (
                    new_word_count > word_target
                    or (max_sentences is not None and new_sentence_count > max_sentences)
                ):
                    flush(s_start)
                    current_start = s_start
                    current_word_count = sent_words
                    current_sentence_count = 1
                else:
                    current_word_count = new_word_count
                    current_sentence_count = new_sentence_count

            if current_sentence_count > 0:
                flush(spans[-1][1])

        return [c for c in chunks if c] or [text]

    def _apply_voice_shaping(self, voice: VoiceType, exaggeration: float) -> float:
        if voice == "female":
            return max(exaggeration, FEMALE_EXAGGERATION_BASE)
        if voice == "male":
            return max(exaggeration, MALE_EXAGGERATION_BASE)
        return exaggeration

    @staticmethod
    def _ensure_mono(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    @staticmethod
    def _fade(wav: torch.Tensor, sr: int, fade_ms: int) -> torch.Tensor:
        if fade_ms <= 0:
            return wav
        n = wav.size(-1)
        fade = int(sr * (fade_ms / 1000.0))
        if fade <= 1 or fade * 2 >= n:
            return wav
        ramp = torch.linspace(0.0, 1.0, fade, device=wav.device, dtype=wav.dtype)
        wav[..., :fade] *= ramp
        wav[..., -fade:] *= ramp.flip(0)
        return wav

    @staticmethod
    def _trim_end(wav: torch.Tensor, sr: int, trim_db: float, pad_ms: int) -> torch.Tensor:
        if wav.numel() == 0:
            return wav
        x = wav[0]
        thr = float(10.0 ** (trim_db / 20.0))
        mask = (x.abs() > thr)
        if not torch.any(mask):
            keep = min(x.numel(), int(sr * 0.25))
            return wav[..., :keep]
        last_idx = int(torch.nonzero(mask, as_tuple=False)[-1].item())
        pad = int(sr * (pad_ms / 1000.0))
        end = min(x.numel(), last_idx + 1 + pad)
        return wav[..., :end]

    def _postprocess(self, wav: torch.Tensor) -> torch.Tensor:
        sr = int(self.model.sr)
        wav = self._ensure_mono(wav).detach()
        wav = wav.clamp(-1.2, 1.2)

        if HIGHPASS_HZ and HIGHPASS_HZ > 0:
            try:
                wav = ta.functional.highpass_biquad(wav, sr, cutoff_freq=float(HIGHPASS_HZ))
            except Exception:
                pass

        if TRIM_END_SILENCE:
            wav = self._trim_end(wav, sr, TRIM_DB, TRIM_PAD_MS)

        wav = self._fade(wav, sr, FADE_MS)
        wav = wav.clamp(-1.0, 1.0)
        return wav

    def _model_generate(self, **kwargs: Any) -> torch.Tensor:
        if self._generate_sig_params is None:
            return self.model.generate(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in self._generate_sig_params}
        return self.model.generate(**filtered)

    @staticmethod
    def _expected_seconds(words: int, speed: float) -> float:
        return max(0.2, (words / 2.6) / max(0.5, float(speed)))

    def _generate_chunk_with_retries(
        self,
        text: str,
        language: str,
        voice: VoiceType,
        temperature: float,
        cfg_weight: float,
        exaggeration: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
        min_new_tokens: int,
        seed: Optional[int],
        speed: float,
    ) -> torch.Tensor:
        audio_prompt = self.voice_manager.get_voice_path(voice)
        exaggeration = self._apply_voice_shaping(voice, exaggeration)

        words = self._approx_word_count(text)
        expected = self._expected_seconds(words, speed)

        ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()

        last_wav: Optional[torch.Tensor] = None
        for attempt in range(MAX_RETRIES_SHORT_AUDIO + 1):
            t = float(temperature)
            cfg = float(cfg_weight)
            rp = float(repetition_penalty)
            tp = float(top_p)

            if attempt > 0:
                t = min(2.0, t * (1.15 + 0.05 * attempt))
                cfg = max(0.0, cfg * 0.85)
                rp = max(1.0, rp * 0.95)
                tp = min(1.0, tp + 0.02 * attempt)

            if seed is not None:
                torch.manual_seed(int(seed) + attempt)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed) + attempt)

            with self._gen_lock:
                with ctx:
                    wav = self._model_generate(
                        text=text,
                        language_id=language,
                        audio_prompt_path=audio_prompt,
                        exaggeration=exaggeration,
                        cfg_weight=cfg,
                        temperature=t,
                        top_p=tp,
                        top_k=top_k,
                        repetition_penalty=rp,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                    )

            wav = self._postprocess(wav)
            sr = int(self.model.sr)
            dur = float(wav.size(-1)) / float(sr) if wav.numel() else 0.0
            last_wav = wav

            min_ok = max(MIN_AUDIO_SEC_FLOOR, expected * MIN_AUDIO_FRAC_OF_EXPECTED)
            if dur >= min_ok:
                return wav

            logger.warning(
                "Short audio detected (%.2fs, expected~%.2fs). Retrying (%d/%d)...",
                dur, expected, attempt + 1, MAX_RETRIES_SHORT_AUDIO
            )

        assert last_wav is not None
        return last_wav

    @staticmethod
    def _to_pcm16le_bytes(wav_mono: torch.Tensor) -> bytes:
        wav_mono = wav_mono.clamp(-1.0, 1.0)
        x = (wav_mono[0] * 32767.0).round().to(torch.int16).cpu().numpy()
        return x.tobytes(order="C")

    def _apply_speed_resample(self, wav_cpu: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
        if speed == 1.0:
            return wav_cpu
        try:
            return ta.functional.resample(wav_cpu, orig_freq=sr, new_freq=int(sr * float(speed)))
        except Exception:
            return wav_cpu

    def _encode_pcm(self, wav_cpu: torch.Tensor) -> bytes:
        return self._to_pcm16le_bytes(wav_cpu)

    def _encode_wav(self, wav_cpu: torch.Tensor, sr: int) -> bytes:
        buf = BytesIO()
        ta.save(buf, wav_cpu, sr, format="wav")
        buf.seek(0)
        return buf.read()

    async def synthesize_streaming(self, req: SpeechRequest) -> AsyncIterator[bytes]:
        self._active_requests += 1
        try:
            word_target = int(req.chunk_words or DEFAULT_CHUNK_WORDS)
            max_sents = req.max_chunk_sentences if req.max_chunk_sentences is not None else DEFAULT_MAX_SENTENCES

            if req.chunk_by_sentences:
                chunks = self._chunk_text(
                    req.input,
                    word_target=word_target,
                    max_sentences=max_sents,
                    preserve_paragraphs=req.preserve_paragraphs,
                )
            else:
                chunks = [req.input]

            num_chunks = len(chunks)
            logger.info("Stream: %d chunks | %s | %s", num_chunks, req.language, req.voice)

            loop = asyncio.get_event_loop()
            sr = int(self.model.sr)

            async def run_one(text: str) -> bytes:
                def _work() -> bytes:
                    wav = self._generate_chunk_with_retries(
                        text=text,
                        language=req.language,
                        voice=req.voice,
                        temperature=req.temperature,
                        cfg_weight=req.cfg_weight,
                        exaggeration=req.exaggeration,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repetition_penalty=req.repetition_penalty,
                        max_new_tokens=req.max_new_tokens,
                        min_new_tokens=req.min_new_tokens,
                        seed=req.seed,
                        speed=req.speed,
                    )
                    wav_cpu = wav.detach().cpu()
                    wav_cpu = self._apply_speed_resample(wav_cpu, sr, req.speed)
                    return self._encode_pcm(wav_cpu) if PCM_STREAM else self._encode_wav(wav_cpu, sr)

                return await loop.run_in_executor(self._executor, _work)

            for chunk_text in chunks:
                yield await run_one(chunk_text)

        except Exception as exc:
            logger.exception("Streaming TTS error")
            raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
        finally:
            self._active_requests -= 1

    async def synthesize(self, req: SpeechRequest) -> bytes:
        """
        Non-streaming synthesis: chunk + concatenate into ONE proper WAV file.
        (Recommended for long blog pages.)
        """
        self._active_requests += 1
        try:
            word_target = int(req.chunk_words or DEFAULT_CHUNK_WORDS)
            max_sents = req.max_chunk_sentences if req.max_chunk_sentences is not None else DEFAULT_MAX_SENTENCES

            if req.chunk_by_sentences:
                chunks = self._chunk_text(
                    req.input,
                    word_target=word_target,
                    max_sentences=max_sents,
                    preserve_paragraphs=req.preserve_paragraphs,
                )
            else:
                chunks = [req.input]

            logger.info("Batch: %d chunks | %s | %s", len(chunks), req.language, req.voice)

            loop = asyncio.get_event_loop()
            sr = int(self.model.sr)
            gap = int(sr * (ADD_GAP_MS / 1000.0))
            gap_wav = torch.zeros(1, gap, dtype=torch.float32)

            def _work() -> bytes:
                parts: list[torch.Tensor] = []
                for i, chunk_text in enumerate(chunks):
                    wav = self._generate_chunk_with_retries(
                        text=chunk_text,
                        language=req.language,
                        voice=req.voice,
                        temperature=req.temperature,
                        cfg_weight=req.cfg_weight,
                        exaggeration=req.exaggeration,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repetition_penalty=req.repetition_penalty,
                        max_new_tokens=req.max_new_tokens,
                        min_new_tokens=req.min_new_tokens,
                        seed=req.seed,
                        speed=req.speed,
                    ).detach().cpu()

                    wav = self._apply_speed_resample(wav, sr, req.speed)
                    parts.append(wav)
                    if i != len(chunks) - 1 and gap > 0:
                        parts.append(gap_wav)

                full = torch.cat(parts, dim=-1) if parts else torch.zeros(1, 1)
                return self._encode_wav(full, sr)

            return await loop.run_in_executor(self._executor, _work)

        except Exception as exc:
            logger.exception("TTS error")
            raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
        finally:
            self._active_requests -= 1


# -----------------------------------------------------------------------------
# FastAPI Application with Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up multilingual TTS server...")
    service: ChatterboxServiceMultilingual = app.state.tts_service
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, service.initialize)
    logger.info("✓ Server ready for requests in %d languages", len(SUPPORTED_LANGUAGES))
    yield
    logger.info("🛶 FastAPI shutdown received, cleaning up TTS service...")
    await loop.run_in_executor(None, service.shutdown)


def create_app(device: str) -> FastAPI:
    app = FastAPI(
        title="VRSecretary Chatterbox TTS - Multilingual",
        version="3.0.1",
        description="Production long-form streaming TTS supporting 23 languages",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = ChatterboxServiceMultilingual(device=device)
    app.state.tts_service = service

    @app.get("/health", response_model=HealthResponse)
    async def health():
        svc: ChatterboxServiceMultilingual = app.state.tts_service
        vm = svc.voice_manager if svc._initialized else None
        return HealthResponse(
            status="ready" if svc._initialized else "initializing",
            device=svc.device,
            model_ready=svc._initialized,
            voices_loaded={
                "female": vm.is_loaded("female") if vm else False,
                "male": vm.is_loaded("male") if vm else False,
            },
            active_requests=svc._active_requests,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    @app.get("/languages", response_model=LanguagesResponse)
    async def get_languages():
        return LanguagesResponse(languages=SUPPORTED_LANGUAGES, count=len(SUPPORTED_LANGUAGES))

    @app.post("/v1/audio/speech/stream")
    async def v1_audio_speech_stream(request: SpeechRequest):
        svc: ChatterboxServiceMultilingual = app.state.tts_service
        sr = int(svc.model.sr)

        async def audio_generator():
            async for chunk in svc.synthesize_streaming(request):
                yield chunk

        if PCM_STREAM:
            return StreamingResponse(
                audio_generator(),
                media_type="audio/L16",
                headers={
                    "X-Voice-Type": request.voice,
                    "X-Language": request.language,
                    "X-Streaming": "true",
                    "X-Audio-Format": "pcm_s16le",
                    "X-Sample-Rate": str(sr),
                    "X-Channels": "1",
                },
            )

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "X-Voice-Type": request.voice,
                "X-Language": request.language,
                "X-Streaming": "true",
                "X-Audio-Format": "wav_chunks",
            },
        )

    @app.post("/v1/audio/speech")
    async def v1_audio_speech(request: SpeechRequest):
        svc: ChatterboxServiceMultilingual = app.state.tts_service
        if request.stream:
            return await v1_audio_speech_stream(request)

        wav_bytes = await svc.synthesize(request)
        return StreamingResponse(
            BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "X-Voice-Type": request.voice,
                "X-Language": request.language,
                "X-Streaming": "false",
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error="http_error", detail=str(exc.detail)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="internal_error", detail="Server error").model_dump(),
        )

    return app


_device_for_app = detect_device()
app = create_app(_device_for_app)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multilingual Chatterbox TTS Server (Long-form, 23 languages)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("VRCB_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VRCB_PORT", "4123")))
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu/mps)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers")
    parser.add_argument(
        "--log-level",
        default=os.getenv("VRCB_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    device = detect_device(args.device)
    logger.info("🎤 Starting multilingual TTS server on %s:%d", args.host, args.port)
    logger.info("💬 Device: %s | Languages: %d", device, len(SUPPORTED_LANGUAGES))
    logger.info("🎚️ Stream mode: %s", "PCM (recommended)" if PCM_STREAM else "WAV-chunks (not recommended)")
    logger.info("🧩 Long-form chunk default: %d words / %d sentences", DEFAULT_CHUNK_WORDS, DEFAULT_MAX_SENTENCES)
    logger.info("🩹 Disable AlignmentStreamAnalyzer: %s", DISABLE_ALIGNMENT_ANALYZER)
    logger.info("🧠 attn_implementation: %s", ATTN_IMPL)

    global app
    if device != _device_for_app:
        app = create_app(device)

    import uvicorn
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            workers=args.workers,
            http="h11",
            timeout_keep_alive=30,
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error("Server crashed: %s", e)


if __name__ == "__main__":
    main()
