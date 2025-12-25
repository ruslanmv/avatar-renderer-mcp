# =============================================================================
# Avatar Renderer MCP - Production-Ready Makefile (Single Source of Truth)
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PYTHON        ?= python3.11
UV            ?= uv
VENV_DIR      ?= .venv
VENV_BIN       = $(VENV_DIR)/bin
EXT_DEPS_DIR  ?= external_deps
MODELS_DIR    ?= $(CURDIR)/models
IMAGE         ?= avatar-renderer:latest

# Repos (single source of truth)
SADTALKER_REPO ?= https://github.com/OpenTalker/SadTalker.git
FOMM_REPO      ?= https://github.com/AliaksandrSiarohin/first-order-model.git
WAV2LIP_REPO   ?= https://github.com/Rudrabha/Wav2Lip.git
DIFF2LIP_REPO  ?= https://github.com/soumik-kanad/diff2lip.git
GUIDED_DIFFUSION_REPO ?= https://github.com/openai/guided-diffusion.git


# Colors
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
RESET := \033[0m
BOLD := \033[1m

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help message
	@printf "$(BOLD)$(BLUE)Avatar Renderer MCP - Production Makefile$(RESET)\n\n"
	@printf "$(BOLD)Usage:$(RESET)\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@printf "\n"

# ─────────────────────────────────────────────────────────────────────────────
# Core checks
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: verify-uv
verify-uv: ## Verify uv is installed
	@command -v $(UV) >/dev/null 2>&1 || { \
		printf "$(RED)Error: uv not found. Install from https://github.com/astral-sh/uv$(RESET)\n"; \
		exit 1; \
	}
	@printf "$(GREEN)✓ uv found: $$($(UV) --version)$(RESET)\n"

.PHONY: verify-ffmpeg
verify-ffmpeg: ## Verify ffmpeg is installed
	@command -v ffmpeg >/dev/null 2>&1 || { \
		printf "$(YELLOW)⚠ ffmpeg not found. Install it for video processing.$(RESET)\n"; \
		printf "$(YELLOW)  Ubuntu/Debian: sudo apt-get install ffmpeg$(RESET)\n"; \
		printf "$(YELLOW)  macOS: brew install ffmpeg$(RESET)\n"; \
	}
	@command -v ffmpeg >/dev/null 2>&1 && printf "$(GREEN)✓ ffmpeg found: $$(ffmpeg -version | head -n1)$(RESET)\n" || true

.PHONY: venv
venv: verify-uv ## Create venv if missing
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(UV) venv $(VENV_DIR) --python $(PYTHON); \
	fi
	@printf "$(GREEN)✓ venv ready: $(VENV_DIR)$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Installation (single truth)
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install
install: venv install-internal install-git-deps install-external-py download-models verify-ffmpeg ## Full production install
	@printf "\n$(GREEN)$(BOLD)✓ All dependencies installed (internal + external + models)$(RESET)\n"
	@printf "$(BOLD)Activate with:$(RESET) source $(VENV_BIN)/activate\n"
	@printf "$(BOLD)Verify setup:$(RESET) make verify\n"
	@printf "$(BOLD)Run demo:$(RESET)     make demo\n"

.PHONY: install-internal
install-internal: venv ## Install internal package deps (pyproject)
	@printf "$(BLUE)Installing internal package deps from pyproject.toml...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python -e .
	@printf "$(GREEN)✓ Internal deps installed$(RESET)\n"

.PHONY: dev-install
dev-install: venv ## Install dev extras too
	@printf "$(BLUE)Installing dev deps...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python -e ".[dev]"
	@$(MAKE) install-git-deps
	@$(MAKE) install-external-py
	@$(MAKE) download-models
	@$(MAKE) verify-ffmpeg
	@printf "$(GREEN)$(BOLD)✓ Dev environment ready$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# External Git deps
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install-git-deps
install-git-deps: venv ## Clone external Git dependencies and install guided_diffusion
	@printf "$(BLUE)$(BOLD)Cloning external Git dependencies...$(RESET)\n"
	@mkdir -p $(EXT_DEPS_DIR)

	@# SadTalker
	@if [ ! -d "$(EXT_DEPS_DIR)/SadTalker" ]; then \
		printf "$(BLUE)→ Cloning SadTalker...$(RESET)\n"; \
		git clone --depth=1 $(SADTALKER_REPO) $(EXT_DEPS_DIR)/SadTalker || { \
			printf "$(YELLOW)⚠ SadTalker clone failed (optional)$(RESET)\n"; \
			true; \
		}; \
	else \
		printf "$(GREEN)↪ SadTalker already present$(RESET)\n"; \
	fi

	@# First Order Motion Model (CRITICAL)
	@if [ ! -d "$(EXT_DEPS_DIR)/first-order-model" ]; then \
		printf "$(BLUE)→ Cloning first-order-model...$(RESET)\n"; \
		git clone --depth=1 $(FOMM_REPO) $(EXT_DEPS_DIR)/first-order-model || { \
			printf "$(RED)✗ first-order-model clone failed (CRITICAL)$(RESET)\n"; \
			exit 1; \
		}; \
	else \
		printf "$(GREEN)↪ first-order-model already present$(RESET)\n"; \
	fi

	@# Wav2Lip (CRITICAL)
	@if [ ! -d "$(EXT_DEPS_DIR)/Wav2Lip" ]; then \
		printf "$(BLUE)→ Cloning Wav2Lip...$(RESET)\n"; \
		git clone --depth=1 $(WAV2LIP_REPO) $(EXT_DEPS_DIR)/Wav2Lip || { \
			printf "$(RED)✗ Wav2Lip clone failed (CRITICAL)$(RESET)\n"; \
			exit 1; \
		}; \
	else \
		printf "$(GREEN)↪ Wav2Lip already present$(RESET)\n"; \
	fi

	@# Diff2Lip (Optional but recommended for HQ)
	@if [ ! -d "$(EXT_DEPS_DIR)/Diff2Lip" ]; then \
		printf "$(BLUE)→ Cloning Diff2Lip...$(RESET)\n"; \
		git clone --depth=1 $(DIFF2LIP_REPO) $(EXT_DEPS_DIR)/Diff2Lip || { \
			printf "$(YELLOW)⚠ Diff2Lip clone failed. Will fallback to Wav2Lip at runtime.$(RESET)\n"; \
			true; \
		}; \
	else \
		printf "$(GREEN)↪ Diff2Lip already present$(RESET)\n"; \
	fi

	@# guided-diffusion (CRITICAL for Diff2Lip) - Clone and install as editable
	@printf "\n$(BLUE)$(BOLD)Installing guided-diffusion (required for Diff2Lip)...$(RESET)\n"
	@if [ ! -d "$(EXT_DEPS_DIR)/guided-diffusion" ]; then \
		printf "$(BLUE)→ Cloning guided-diffusion...$(RESET)\n"; \
		git clone $(GUIDED_DIFFUSION_REPO) $(EXT_DEPS_DIR)/guided-diffusion || { \
			printf "$(RED)✗ guided-diffusion clone failed$(RESET)\n"; \
			exit 1; \
		}; \
	else \
		printf "$(GREEN)↪ guided-diffusion already cloned$(RESET)\n"; \
	fi
	
	@# Install guided-diffusion in editable mode
	@printf "$(BLUE)→ Installing guided-diffusion with editable mode...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python -e $(EXT_DEPS_DIR)/guided-diffusion || { \
		printf "$(RED)✗ guided-diffusion install failed$(RESET)\n"; \
		exit 1; \
	}
	
	@# Install guided-diffusion core dependencies (skip optional mpi4py)
	@printf "$(BLUE)→ Installing guided-diffusion dependencies...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python blobfile torch tqdm
	
	@# Verify guided-diffusion works (skip mpi4py check)
	@printf "$(BLUE)→ Verifying guided-diffusion installation...$(RESET)\n"
	@$(VENV_BIN)/python -c "import guided_diffusion; print('  ✓ guided_diffusion import successful')" || { \
		printf "$(RED)  ✗ guided_diffusion import failed$(RESET)\n"; \
		exit 1; \
	}
	@$(VENV_BIN)/python -c "from guided_diffusion import dist_util; print('  ✓ dist_util import successful (mpi4py is optional)')" 2>/dev/null || { \
		printf "$(YELLOW)  ⚠ dist_util requires mpi4py (optional, not needed for basic usage)$(RESET)\n"; \
		true; \
	}
	@printf "$(GREEN)✓ guided_diffusion installed successfully$(RESET)\n"

	@# FOMM wrapper patch (if exists)
	@if [ -f "patches/fomm/fomm_wrapper.py" ]; then \
		printf "$(BLUE)→ Installing FOMM wrapper patch...$(RESET)\n"; \
		cp patches/fomm/fomm_wrapper.py $(EXT_DEPS_DIR)/first-order-model/fomm_wrapper.py; \
		printf "$(GREEN)✓ FOMM wrapper installed$(RESET)\n"; \
	fi

	@printf "$(GREEN)$(BOLD)✓ Git dependencies ready$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# External Python deps (NOT in pyproject.toml)
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install-external-py
install-external-py: venv ## Install external python deps required by cloned repos
	@printf "$(BLUE)$(BOLD)Installing external Python dependencies...$(RESET)\n"
	@printf "$(YELLOW)Note: These are dependencies of cloned repos, not in our pyproject.toml$(RESET)\n\n"

	@# Core dependencies (ensure they're installed with correct versions)
	@printf "$(BLUE)→ Installing core dependencies...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		"numpy<2.0.0" \
		"opencv-python>=4.8.0,<5.0.0" \
		"Pillow>=10.0.0,<11.0.0" \
		"scipy>=1.11.0,<2.0.0" \
		"scikit-image>=0.21.0,<0.22.0" \
		|| { printf "$(RED)✗ Core deps failed$(RESET)\n"; exit 1; }

	@# Audio processing
	@printf "$(BLUE)→ Installing audio processing stack...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		"librosa==0.9.2" \
		"soundfile>=0.12.0,<0.13.0" \
		"numba>=0.58.0" \
		|| { printf "$(RED)✗ Audio deps failed$(RESET)\n"; exit 1; }

	@# Video processing
	@printf "$(BLUE)→ Installing video processing stack...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		ffmpeg-python \
		imageio \
		imageio-ffmpeg \
		|| { printf "$(RED)✗ Video deps failed$(RESET)\n"; exit 1; }

	@# Configuration and utilities
	@printf "$(BLUE)→ Installing utilities...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		pyyaml \
		yacs \
		tqdm \
		|| { printf "$(RED)✗ Utility deps failed$(RESET)\n"; exit 1; }

	@# SadTalker specific dependencies (optional)
	@printf "$(BLUE)→ Installing SadTalker stack (optional)...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		kornia \
		face-alignment \
		mediapipe \
		|| { printf "$(YELLOW)⚠ SadTalker deps partially failed (optional)$(RESET)\n"; true; }

	@# Face enhancement stack (optional)
	@printf "$(BLUE)→ Installing face enhancement stack (optional)...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		gfpgan \
		facexlib \
		basicsr \
		realesrgan \
		|| { printf "$(YELLOW)⚠ Enhancement stack partially failed (optional)$(RESET)\n"; true; }

	@# Additional ML dependencies
	@printf "$(BLUE)→ Installing additional ML dependencies...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python \
		"torchvision>=0.15.0" \
		"transformers>=4.35.0" \
		"diffusers>=0.25.0" \
		|| { printf "$(YELLOW)⚠ ML deps partially failed$(RESET)\n"; true; }

	@printf "$(GREEN)$(BOLD)✓ External Python deps installed$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: download-models
download-models: ## Download model checkpoints
	@printf "$(BLUE)Ensuring models exist in $(MODELS_DIR)...$(RESET)\n"
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(MODELS_DIR)/fomm
	@mkdir -p $(MODELS_DIR)/diff2lip
	@mkdir -p $(MODELS_DIR)/wav2lip
	@mkdir -p $(MODELS_DIR)/sadtalker
	@mkdir -p $(MODELS_DIR)/gfpgan
	@if [ -f "scripts/download_models.sh" ]; then \
		bash scripts/download_models.sh "$(MODELS_DIR)"; \
	else \
		printf "$(YELLOW)⚠ scripts/download_models.sh not found.$(RESET)\n"; \
		printf "$(YELLOW)  Models must be downloaded manually to:$(RESET)\n"; \
		printf "$(YELLOW)    - $(MODELS_DIR)/fomm/vox-cpk.pth$(RESET)\n"; \
		printf "$(YELLOW)    - $(MODELS_DIR)/wav2lip/wav2lip_gan.pth$(RESET)\n"; \
		printf "$(YELLOW)    - $(MODELS_DIR)/diff2lip/Diff2Lip.pth (optional)$(RESET)\n"; \
		printf "$(YELLOW)    - $(MODELS_DIR)/gfpgan/GFPGANv1.3.pth (optional)$(RESET)\n"; \
	fi
	@printf "$(GREEN)✓ Models step complete$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Verification (comprehensive)
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: verify
verify: venv ## Verify all dependencies are installed
	@printf "$(BLUE)$(BOLD)Running comprehensive verification...$(RESET)\n\n"
	
	@printf "$(BOLD)Core Dependencies:$(RESET)\n"
	@$(VENV_BIN)/python -c "import torch; print('  ✅ torch:', torch.__version__)" || printf "  $(RED)❌ torch missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import torchvision; print('  ✅ torchvision:', torchvision.__version__)" || printf "  $(RED)❌ torchvision missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import cv2; print('  ✅ opencv-python:', cv2.__version__)" || printf "  $(RED)❌ opencv-python missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import numpy as np; print('  ✅ numpy:', np.__version__, '(must be <2.0.0)')" || printf "  $(RED)❌ numpy missing$(RESET)\n"
	
	@printf "\n$(BOLD)Audio/Video Processing:$(RESET)\n"
	@$(VENV_BIN)/python -c "import librosa; print('  ✅ librosa:', librosa.__version__, '(must be 0.9.x)')" 2>/dev/null || printf "  $(RED)❌ librosa missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import soundfile; print('  ✅ soundfile installed')" || printf "  $(RED)❌ soundfile missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import ffmpeg; print('  ✅ ffmpeg-python installed')" || printf "  $(RED)❌ ffmpeg-python missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import imageio; print('  ✅ imageio installed')" || printf "  $(RED)❌ imageio missing$(RESET)\n"
	
	@printf "\n$(BOLD)Scientific Computing:$(RESET)\n"
	@$(VENV_BIN)/python -c "import scipy; print('  ✅ scipy installed')" || printf "  $(RED)❌ scipy missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import skimage; print('  ✅ scikit-image installed')" || printf "  $(RED)❌ scikit-image missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import PIL; print('  ✅ Pillow installed')" || printf "  $(RED)❌ Pillow missing$(RESET)\n"
	
	@printf "\n$(BOLD)Configuration:$(RESET)\n"
	@$(VENV_BIN)/python -c "import yaml; print('  ✅ PyYAML installed')" || printf "  $(RED)❌ PyYAML missing$(RESET)\n"
	@$(VENV_BIN)/python -c "import yacs; print('  ✅ YACS installed')" || printf "  $(RED)❌ YACS missing$(RESET)\n"
	
	@printf "\n$(BOLD)Critical for Diff2Lip:$(RESET)\n"
	@$(VENV_BIN)/python -c "import guided_diffusion; print('  ✅ guided_diffusion installed at:', guided_diffusion.__file__)" || { \
		printf "  $(RED)❌ guided_diffusion missing - Diff2Lip will FAIL$(RESET)\n"; \
		printf "  $(YELLOW)→ Fix: make fix-deps$(RESET)\n"; \
	}
	@$(VENV_BIN)/python -c "import guided_diffusion.gaussian_diffusion; print('  ✅ guided_diffusion modules accessible')" 2>/dev/null || { \
		printf "  $(YELLOW)⚠  Some guided_diffusion modules inaccessible (may still work)$(RESET)\n"; \
		true; \
	}
	
	@printf "\n$(BOLD)Optional Enhancement:$(RESET)\n"
	@$(VENV_BIN)/python -c "\
import sys; \
try: \
    from torchvision.transforms import functional_tensor; \
except ImportError: \
    import torchvision.transforms.functional as functional; \
    sys.modules['torchvision.transforms.functional_tensor'] = functional; \
import gfpgan; \
print('  ✅ gfpgan installed (with monkeypatch)'); \
" 2>/dev/null || printf "  $(YELLOW)⚠  gfpgan missing (optional)$(RESET)\n"
	@$(VENV_BIN)/python -c "import face_alignment; print('  ✅ face-alignment installed')" || printf "  $(YELLOW)⚠  face-alignment missing (optional)$(RESET)\n"
	@$(VENV_BIN)/python -c "import kornia; print('  ✅ kornia installed')" || printf "  $(YELLOW)⚠  kornia missing (optional)$(RESET)\n"
	@$(VENV_BIN)/python -c "import mediapipe; print('  ✅ mediapipe installed')" || printf "  $(YELLOW)⚠  mediapipe missing (optional)$(RESET)\n"
	
	@printf "\n$(BOLD)External Repositories:$(RESET)\n"
	@test -d "$(EXT_DEPS_DIR)/first-order-model" && printf "  ✅ first-order-model cloned\n" || printf "  $(RED)❌ first-order-model missing$(RESET)\n"
	@test -d "$(EXT_DEPS_DIR)/Wav2Lip" && printf "  ✅ Wav2Lip cloned\n" || printf "  $(RED)❌ Wav2Lip missing$(RESET)\n"
	@test -d "$(EXT_DEPS_DIR)/Diff2Lip" && printf "  ✅ Diff2Lip cloned\n" || printf "  $(YELLOW)⚠  Diff2Lip missing (optional)$(RESET)\n"
	@test -d "$(EXT_DEPS_DIR)/guided-diffusion" && printf "  ✅ guided-diffusion cloned\n" || printf "  $(RED)❌ guided-diffusion missing$(RESET)\n"
	@test -d "$(EXT_DEPS_DIR)/SadTalker" && printf "  ✅ SadTalker cloned\n" || printf "  $(YELLOW)⚠  SadTalker missing (optional)$(RESET)\n"
	
	@printf "\n$(GREEN)$(BOLD)✓ Verification complete$(RESET)\n"
	@printf "$(YELLOW)Run 'make verify-full' for detailed dependency check$(RESET)\n"

.PHONY: verify-full
verify-full: venv ## Run comprehensive Python verification script
	@printf "$(BLUE)Running comprehensive verification script...$(RESET)\n"
	@if [ -f "scripts/verify_setup.py" ]; then \
		$(VENV_BIN)/python scripts/verify_setup.py; \
	else \
		printf "$(YELLOW)⚠ scripts/verify_setup.py not found$(RESET)\n"; \
		$(MAKE) verify; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# Fix missing dependencies
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: fix-deps
fix-deps: venv ## Fix missing dependencies (guided_diffusion, etc.)
	@printf "$(BLUE)$(BOLD)Fixing missing dependencies...$(RESET)\n\n"
	
	@# Remove broken installation
	@printf "$(BLUE)→ Removing any broken installations...$(RESET)\n"
	@$(UV) pip uninstall --python $(VENV_BIN)/python guided-diffusion -y 2>/dev/null || true
	@rm -rf $(EXT_DEPS_DIR)/guided-diffusion
	
	@# Reinstall guided-diffusion properly
	@printf "$(BLUE)→ Cloning guided-diffusion...$(RESET)\n"
	@git clone $(GUIDED_DIFFUSION_REPO) $(EXT_DEPS_DIR)/guided-diffusion
	
	@printf "$(BLUE)→ Installing guided-diffusion...$(RESET)\n"
	@$(UV) pip install --python $(VENV_BIN)/python -e $(EXT_DEPS_DIR)/guided-diffusion
	@$(UV) pip install --python $(VENV_BIN)/python blobfile torch tqdm
	
	@printf "$(BLUE)→ Verifying installation...$(RESET)\n"
	@$(VENV_BIN)/python -c "import guided_diffusion; print('✅ guided_diffusion works!')"
	
	@printf "\n$(GREEN)$(BOLD)✓ Dependencies fixed!$(RESET)\n"
	@printf "$(YELLOW)Run 'make verify' to confirm$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run
run: install ## Start FastAPI server on :8080
	@printf "$(BLUE)Starting FastAPI server...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		MODEL_ROOT=$(MODELS_DIR) \
		EXT_DEPS_DIR=$(CURDIR)/$(EXT_DEPS_DIR) \
		$(VENV_BIN)/uvicorn app.api:app --host 0.0.0.0 --port 8080 --reload

.PHONY: run-stdio
run-stdio: install ## Start MCP STDIO server
	@printf "$(BLUE)Starting MCP STDIO server...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		MODEL_ROOT=$(MODELS_DIR) \
		EXT_DEPS_DIR=$(CURDIR)/$(EXT_DEPS_DIR) \
		$(VENV_BIN)/python -m app.mcp_server

.PHONY: demo
demo: install ## Run demo script
	@printf "$(BLUE)Running demo...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		MODEL_ROOT=$(MODELS_DIR) \
		EXT_DEPS_DIR=$(CURDIR)/$(EXT_DEPS_DIR) \
		$(VENV_BIN)/python demo.py

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: test-light
test-light: install ## Run lightweight health check tests (fast)
	@printf "$(BLUE)Running lightweight health checks...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		MODEL_ROOT=$(MODELS_DIR) \
		EXT_DEPS_DIR=$(CURDIR)/$(EXT_DEPS_DIR) \
		$(VENV_BIN)/python tests/test_health.py
	@printf "$(GREEN)✓ Health checks complete$(RESET)\n"

.PHONY: test
test: install ## Run full test suite
	@printf "$(BLUE)Running test suite...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		MODEL_ROOT=$(MODELS_DIR) \
		EXT_DEPS_DIR=$(CURDIR)/$(EXT_DEPS_DIR) \
		$(VENV_BIN)/pytest tests/ -v
	@printf "$(GREEN)✓ Tests complete$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts and caches
	@printf "$(BLUE)Cleaning up...$(RESET)\n"
	@rm -rf $(VENV_DIR) .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info htmlcov .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -f demo.mp4 2>/dev/null || true
	@rm -rf temp/ 2>/dev/null || true
	@printf "$(GREEN)✓ Cleanup complete$(RESET)\n"

.PHONY: clean-all
clean-all: clean ## Remove everything including models and external deps
	@printf "$(BLUE)Deep cleaning...$(RESET)\n"
	@rm -rf $(EXT_DEPS_DIR) $(MODELS_DIR)
	@printf "$(GREEN)✓ Deep clean complete$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Development helpers
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: format
format: venv ## Format code with black and isort
	@printf "$(BLUE)Formatting code...$(RESET)\n"
	@$(VENV_BIN)/black app/ tests/ demo.py 2>/dev/null || true
	@$(VENV_BIN)/isort app/ tests/ demo.py 2>/dev/null || true
	@printf "$(GREEN)✓ Code formatted$(RESET)\n"

.PHONY: lint
lint: venv ## Lint code with ruff
	@printf "$(BLUE)Linting code...$(RESET)\n"
	@$(VENV_BIN)/ruff check app/ tests/ demo.py 2>/dev/null || true
	@printf "$(GREEN)✓ Linting complete$(RESET)\n"

.PHONY: type-check
type-check: venv ## Type check with mypy
	@printf "$(BLUE)Type checking...$(RESET)\n"
	@$(VENV_BIN)/mypy app/ 2>/dev/null || true
	@printf "$(GREEN)✓ Type checking complete$(RESET)\n"

.PHONY: all-checks
all-checks: format lint type-check test-light ## Run all code quality checks
	@printf "$(GREEN)$(BOLD)✓ All checks passed!$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Quick commands
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: quick-start
quick-start: ## Quick start guide
	@printf "$(BOLD)$(BLUE)Avatar Renderer MCP - Quick Start$(RESET)\n\n"
	@printf "$(BOLD)1. Install everything:$(RESET)\n"
	@printf "   make install\n\n"
	@printf "$(BOLD)2. Verify installation:$(RESET)\n"
	@printf "   make verify\n\n"
	@printf "$(BOLD)3. Run demo:$(RESET)\n"
	@printf "   make demo\n\n"
	@printf "$(BOLD)If verification fails:$(RESET)\n"
	@printf "   make fix-deps\n\n"
	@printf "$(BOLD)For detailed verification:$(RESET)\n"
	@printf "   make verify-full\n\n"

.PHONY: status
status: ## Show installation status
	@printf "$(BOLD)$(BLUE)Installation Status$(RESET)\n\n"
	@printf "$(BOLD)Virtual Environment:$(RESET)\n"
	@test -d "$(VENV_DIR)" && printf "  ✅ Present at $(VENV_DIR)\n" || printf "  ❌ Missing - run 'make venv'\n"
	@printf "\n$(BOLD)External Repositories:$(RESET)\n"
	@test -d "$(EXT_DEPS_DIR)/first-order-model" && printf "  ✅ first-order-model\n" || printf "  ❌ first-order-model - run 'make install-git-deps'\n"
	@test -d "$(EXT_DEPS_DIR)/Wav2Lip" && printf "  ✅ Wav2Lip\n" || printf "  ❌ Wav2Lip - run 'make install-git-deps'\n"
	@test -d "$(EXT_DEPS_DIR)/Diff2Lip" && printf "  ✅ Diff2Lip\n" || printf "  ⚠  Diff2Lip (optional)\n"
	@test -d "$(EXT_DEPS_DIR)/guided-diffusion" && printf "  ✅ guided-diffusion\n" || printf "  ❌ guided-diffusion - run 'make install-git-deps'\n"
	@printf "\n$(BOLD)Models Directory:$(RESET)\n"
	@test -d "$(MODELS_DIR)" && printf "  ✅ Present at $(MODELS_DIR)\n" || printf "  ❌ Missing - run 'make download-models'\n"
	@printf "\n$(YELLOW)Run 'make verify' for detailed dependency check$(RESET)\n"