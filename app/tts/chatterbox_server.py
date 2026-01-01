from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from typing import AsyncIterator, Literal, Optional

# -----------------------------------------------------------------------------
# WARNING FILTERS (Clean up logs)
# -----------------------------------------------------------------------------
# Filter annoying library warnings specific to Transfromers/Diffusers/Torch
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
# Assuming this exists in your python path
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    print("CRITICAL ERROR: Could not import 'chatterbox.mtl_tts'. Ensure your python path is correct.")
    exit(1)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger("vr_chatterbox_server_multilingual")

# Supported languages (23 total)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

# Project base / voices dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(BASE_DIR, "voices")

# Performance tuning
MAX_WORKERS = int(os.getenv("CHATTERBOX_MAX_WORKERS", "4"))
FAST_MODE = os.getenv("CHATTERBOX_FAST_MODE", "true").lower() == "true"
DEFAULT_CHUNK_SIZE = int(os.getenv("CHATTERBOX_CHUNK_SIZE", "15"))
FAST_CHUNK_SIZE = int(os.getenv("CHATTERBOX_FAST_CHUNK_SIZE", "10"))
CHUNK_SIZE = FAST_CHUNK_SIZE if FAST_MODE else DEFAULT_CHUNK_SIZE

# Voice fine-tuning
FEMALE_EXAGGERATION_BASE = float(os.getenv("CHATTERBOX_FEMALE_EXAGGERATION", "0.45"))
MALE_EXAGGERATION_BASE = float(os.getenv("CHATTERBOX_MALE_EXAGGERATION", "0.30"))

# Sentence splitting
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[\.!?。！？])\s+")


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

    input: str = Field(..., description="Text to synthesize.", min_length=1, max_length=5000)
    language: LanguageCode = Field("en", description="Language code for synthesis.")
    voice: VoiceType = Field("neutral", description="Voice profile: 'female', 'male', or 'neutral'.")
    
    # Generation parameters
    temperature: float = Field(0.7, ge=0.1, le=1.5, description="Sampling temperature.")
    cfg_weight: float = Field(0.4, ge=0.0, le=2.0, description="Guidance weight.")
    exaggeration: float = Field(0.3, ge=0.0, le=1.0, description="Expressiveness level.")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier.")
    
    # Advanced Sampling
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    top_k: int = Field(50, ge=0, description="Top-K sampling.")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Penalty for repetition.")

    # Streaming options
    stream: bool = Field(True, description="Enable streaming.")
    chunk_by_sentences: bool = Field(True, description="Split long text by sentences.")
    max_chunk_words: Optional[int] = Field(None, ge=5, le=2000, description="Max words per chunk.")
    max_chunk_sentences: Optional[int] = Field(None, ge=1, le=200, description="Max sentences per chunk.")


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

        # Ensure directory exists
        if not os.path.exists(VOICES_DIR):
            try:
                os.makedirs(VOICES_DIR, exist_ok=True)
                # Only log if we just created it
            except OSError:
                pass

        # Default paths relative to app/voices/
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

            # Check existence
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
                # Only log warning if path was actually set/defaulted
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
        
        # Ensure file exists before returning it to avoid crashes
        if path and os.path.exists(path):
            return path
        
        return None

    def is_loaded(self, voice: VoiceType) -> bool:
        """Check if a voice profile is loaded."""
        if voice == "neutral":
            return True
        return self.prepared.get(voice, False)


# -----------------------------------------------------------------------------
# Multilingual TTS Service
# -----------------------------------------------------------------------------


class ChatterboxServiceMultilingual:
    """High-performance multilingual TTS service with streaming."""

    def __init__(self, device: str):
        self.device = device
        self._model: Optional[ChatterboxMultilingualTTS] = None
        self._voice_manager: Optional[VoiceProfileManager] = None
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="tts-worker")
        self._active_requests = 0
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._initialized = False

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
                # Load MULTILINGUAL model
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

                if hasattr(self._model, "eval"):
                    self._model.eval()

                torch.set_grad_enabled(False)

                # Load voice profiles
                self._voice_manager = VoiceProfileManager(self._model)
                self._voice_manager.load_all()

                self._initialized = True
                elapsed = time.time() - start
                logger.info("✓ ChatterboxMultilingualTTS ready in %.2f seconds", elapsed)
                logger.info("📚 Supported languages: %d", len(SUPPORTED_LANGUAGES))
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise e

    def shutdown(self) -> None:
        """Cleanly shutdown internal resources."""
        logger.info("🛑 Shutting down TTS executor...")
        self._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("🛑 TTS executor shutdown complete.")

    @property
    def model(self) -> ChatterboxMultilingualTTS:
        if not self._initialized or self._model is None:
            # Fallback initialization
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
        max_sentences: Optional[int] = None,
    ) -> list[str]:
        spans = self._split_into_sentences(text)
        if not spans:
            return [text]

        chunks: list[str] = []
        current_start: Optional[int] = None
        current_word_count = 0
        current_sentence_count = 0

        def flush(end_index: int) -> None:
            nonlocal current_start, current_word_count, current_sentence_count
            if current_start is not None and end_index > current_start:
                chunks.append(text[current_start:end_index])
            current_start = None
            current_word_count = 0
            current_sentence_count = 0

        for s_start, s_end in spans:
            sent_text = text[s_start:s_end]
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
            last_end = spans[-1][1]
            flush(last_end)

        return chunks or [text]

    def _apply_voice_shaping(self, voice: VoiceType, exaggeration: float) -> float:
        if voice == "female":
            return max(exaggeration, FEMALE_EXAGGERATION_BASE)
        if voice == "male":
            return max(exaggeration, MALE_EXAGGERATION_BASE)
        return exaggeration

    def _generate_chunk(
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
    ) -> torch.Tensor:
        """Generate audio for a single text chunk with language support."""
        audio_prompt = self.voice_manager.get_voice_path(voice)
        
        if voice != "neutral" and audio_prompt is None:
            logger.debug(f"Voice '{voice}' requested but file missing. Using Neutral.")

        exaggeration = self._apply_voice_shaping(voice, exaggeration)

        # Handle torch inference context safely
        ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
        
        with self._gen_lock:
            with ctx:
                wav = self.model.generate(
                    text,
                    language_id=language,
                    audio_prompt_path=audio_prompt, 
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )

        return wav

    def _generate_and_encode_chunk(
        self,
        chunk_index: int,
        total_chunks: int,
        text: str,
        language: str,
        voice: VoiceType,
        temperature: float,
        cfg_weight: float,
        exaggeration: float,
        speed: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> bytes:
        chunk_start = time.time()
        
        wav = self._generate_chunk(
            text, language, voice, temperature, cfg_weight, 
            exaggeration, top_p, top_k, repetition_penalty
        )

        buf = BytesIO()
        wav_cpu = wav.detach().cpu()

        if speed != 1.0:
            wav_cpu = ta.functional.resample(
                wav_cpu,
                orig_freq=int(self.model.sr),
                new_freq=int(self.model.sr * speed),
            )

        ta.save(buf, wav_cpu, self.model.sr, format="wav")
        buf.seek(0)

        chunk_bytes = buf.read()
        elapsed = time.time() - chunk_start
        logger.debug(
            "Chunk %d/%d [%s] generated in %.3f seconds (%d bytes)",
            chunk_index + 1,
            total_chunks,
            language,
            elapsed,
            len(chunk_bytes),
        )

        return chunk_bytes

    async def synthesize_streaming(
        self,
        req: SpeechRequest,
    ) -> AsyncIterator[bytes]:
        self._active_requests += 1
        try:
            if req.chunk_by_sentences:
                total_words = self._approx_word_count(req.input)

                if req.max_chunk_words is not None:
                    base_target = req.max_chunk_words
                    effective_target = base_target
                    should_chunk = total_words > base_target
                else:
                    base_target = CHUNK_SIZE
                    effective_target = max(base_target, 20) 
                    should_chunk = total_words > 40

                if should_chunk:
                    word_target = effective_target
                    max_sentences = req.max_chunk_sentences
                    chunks = self._chunk_text(
                        req.input,
                        word_target=word_target,
                        max_sentences=max_sentences,
                    )
                else:
                    chunks = [req.input]
            else:
                chunks = [req.input]

            num_chunks = len(chunks)

            logger.info(
                "Stream: %d chunks | %s | %s",
                num_chunks,
                req.language,
                req.voice,
            )

            loop = asyncio.get_event_loop()

            for idx, chunk_text in enumerate(chunks):
                chunk_bytes = await loop.run_in_executor(
                    self._executor,
                    self._generate_and_encode_chunk,
                    idx,
                    num_chunks,
                    chunk_text,
                    req.language,
                    req.voice,
                    req.temperature,
                    req.cfg_weight,
                    req.exaggeration,
                    req.speed,
                    req.top_p,
                    req.top_k,
                    req.repetition_penalty,
                )
                yield chunk_bytes

        except Exception as exc:
            logger.exception("Streaming TTS error")
            raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
        finally:
            self._active_requests -= 1

    async def synthesize(self, req: SpeechRequest) -> bytes:
        self._active_requests += 1
        try:
            logger.info(
                "Batch: len=%d | %s | %s",
                len(req.input),
                req.language,
                req.voice,
            )

            loop = asyncio.get_event_loop()
            wav_bytes = await loop.run_in_executor(
                self._executor,
                self._generate_and_encode_chunk,
                0,
                1,
                req.input,
                req.language,
                req.voice,
                req.temperature,
                req.cfg_weight,
                req.exaggeration,
                req.speed,
                req.top_p,
                req.top_k,
                req.repetition_penalty,
            )

            return wav_bytes

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
    """
    New Lifespan Context Manager.
    Replaces startup/shutdown events to fix asyncio.CancelledError on reloads.
    """
    logger.info("🚀 Starting up multilingual TTS server...")
    service: ChatterboxServiceMultilingual = app.state.tts_service
    
    # Initialize the heavy model in a separate thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, service.initialize)
    logger.info("✓ Server ready for requests in %d languages", len(SUPPORTED_LANGUAGES))
    
    yield
    
    # Shutdown logic
    logger.info("🛶 FastAPI shutdown received, cleaning up TTS service...")
    await loop.run_in_executor(None, service.shutdown)


def create_app(device: str) -> FastAPI:
    """Create multilingual FastAPI application."""
    app = FastAPI(
        title="VRSecretary Chatterbox TTS - Multilingual",
        version="2.1.2",
        description="High-performance streaming TTS supporting 23 languages",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize service instance (but model loads in lifespan)
    service = ChatterboxServiceMultilingual(device=device)
    app.state.tts_service = service

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check with language support info."""
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
        """Get list of all supported languages."""
        return LanguagesResponse(
            languages=SUPPORTED_LANGUAGES,
            count=len(SUPPORTED_LANGUAGES),
        )

    @app.post("/v1/audio/speech/stream")
    async def v1_audio_speech_stream(request: SpeechRequest):
        """Streaming TTS endpoint with multilingual support."""
        svc: ChatterboxServiceMultilingual = app.state.tts_service

        async def audio_generator():
            async for chunk in svc.synthesize_streaming(request):
                yield chunk

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "X-Voice-Type": request.voice,
                "X-Language": request.language,
                "X-Streaming": "true",
            },
        )

    @app.post("/v1/audio/speech")
    async def v1_audio_speech(request: SpeechRequest):
        """Standard TTS endpoint with multilingual support."""
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
        description="Multilingual Chatterbox TTS Server (23 languages)",
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

    global app
    # Only recreate if device changed from global default, otherwise use existing
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
        logger.error(f"Server crashed: {e}")


if __name__ == "__main__":
    main()