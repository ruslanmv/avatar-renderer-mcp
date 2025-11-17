# =============================================================================
# Avatar Renderer MCP - Production-Ready Makefile
# =============================================================================
# Author: Ruslan Magana Vsevolodovna
# Website: https://ruslanmv.com
# License: Apache-2.0
# Package Manager: uv (astral-sh)
# =============================================================================
#
# Quick Start:
#   make help          # Show all available commands
#   make install       # Install dependencies with uv
#   make run           # Start the server
#
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PYTHON ?= python3.11
UV ?= uv
VENV_DIR ?= .venv
VENV_BIN = $(VENV_DIR)/bin
EXT_DEPS_DIR ?= external_deps
MODELS_DIR ?= $(CURDIR)/models
IMAGE ?= avatar-renderer:latest

# Colors
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
RESET := \033[0m
BOLD := \033[1m

# ─────────────────────────────────────────────────────────────────────────────
# Help (Self-Documenting)
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@printf "$(BOLD)$(BLUE)Avatar Renderer MCP - Production Makefile$(RESET)\n\n"
	@printf "$(BOLD)Usage:$(RESET)\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@printf "\n"

# ─────────────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: verify-uv
verify-uv: ## Verify uv is installed
	@command -v $(UV) >/dev/null 2>&1 || { \
		printf "$(RED)Error: uv not found. Install from https://github.com/astral-sh/uv$(RESET)\n"; \
		exit 1; \
	}
	@printf "$(GREEN)✓ uv found: $$($(UV) --version)$(RESET)\n"

.PHONY: install
install: verify-uv ## Install production dependencies using uv
	@printf "$(BLUE)Installing dependencies with uv...$(RESET)\n"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(UV) venv $(VENV_DIR) --python $(PYTHON); \
	fi
	$(UV) pip install --python $(VENV_BIN)/python -e .
	@printf "$(GREEN)✓ Installation complete$(RESET)\n"
	@printf "$(BOLD)Activate with:$(RESET) source $(VENV_BIN)/activate\n"

.PHONY: dev-install
dev-install: verify-uv ## Install with development dependencies
	@printf "$(BLUE)Installing dev dependencies with uv...$(RESET)\n"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(UV) venv $(VENV_DIR) --python $(PYTHON); \
	fi
	$(UV) pip install --python $(VENV_BIN)/python -e ".[dev]"
	@$(MAKE) install-git-deps
	@$(MAKE) download-models
	@printf "$(GREEN)✓ Dev environment ready$(RESET)\n"

.PHONY: install-git-deps
install-git-deps: ## Clone external Git dependencies
	@printf "$(BLUE)Installing Git dependencies...$(RESET)\n"
	@mkdir -p $(EXT_DEPS_DIR)
	@if [ ! -d "$(EXT_DEPS_DIR)/SadTalker" ]; then \
		git clone --depth=1 https://github.com/OpenTalker/SadTalker.git $(EXT_DEPS_DIR)/SadTalker; \
	fi
	@if [ -f "$(EXT_DEPS_DIR)/SadTalker/requirements.txt" ]; then \
		$(UV) pip install --python $(VENV_BIN)/python -r $(EXT_DEPS_DIR)/SadTalker/requirements.txt; \
	fi
	@if [ ! -d "$(EXT_DEPS_DIR)/first-order-model" ]; then \
		git clone --depth=1 https://github.com/AliaksandrSiarohin/first-order-model.git $(EXT_DEPS_DIR)/first-order-model; \
	fi
	@if [ ! -d "$(EXT_DEPS_DIR)/Wav2Lip" ]; then \
		git clone --depth=1 https://github.com/Rudrabha/Wav2Lip.git $(EXT_DEPS_DIR)/Wav2Lip; \
	fi
	@if [ -f "$(EXT_DEPS_DIR)/Wav2Lip/requirements.txt" ]; then \
		$(UV) pip install --python $(VENV_BIN)/python -r $(EXT_DEPS_DIR)/Wav2Lip/requirements.txt; \
	fi
	@printf "$(GREEN)✓ Git dependencies installed$(RESET)\n"

.PHONY: download-models
download-models: ## Download model checkpoints
	@printf "$(BLUE)Downloading models to $(MODELS_DIR)...$(RESET)\n"
	@mkdir -p $(MODELS_DIR)
	@if [ -f "scripts/download_models.sh" ]; then \
		bash scripts/download_models.sh "$(MODELS_DIR)"; \
	fi
	@printf "$(GREEN)✓ Models downloaded$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Development
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: run
run: install ## Start FastAPI server on :8080
	@printf "$(BLUE)Starting FastAPI server...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/uvicorn app.api:app --host 0.0.0.0 --port 8080 --reload

.PHONY: run-stdio
run-stdio: install ## Start MCP STDIO server
	@printf "$(BLUE)Starting MCP STDIO server...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/python -m app.mcp_server

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test
test: dev-install ## Run test suite with coverage
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/pytest -v
	@printf "$(GREEN)✓ Tests complete$(RESET)\n"

.PHONY: test-cov
test-cov: dev-install ## Run tests with coverage report
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/pytest -v --cov=app --cov-report=html --cov-report=term

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: lint
lint: dev-install ## Run linters (ruff + mypy)
	@printf "$(BLUE)Running linters...$(RESET)\n"
	@$(VENV_BIN)/ruff check app tests scripts || true
	@$(VENV_BIN)/mypy app --ignore-missing-imports || true
	@$(VENV_BIN)/black --check app tests scripts || true
	@printf "$(GREEN)✓ Linting complete$(RESET)\n"

.PHONY: format
format: dev-install ## Format code with black and isort
	@printf "$(BLUE)Formatting code...$(RESET)\n"
	@$(VENV_BIN)/isort app tests scripts
	@$(VENV_BIN)/black app tests scripts
	@printf "$(GREEN)✓ Code formatted$(RESET)\n"

.PHONY: fmt
fmt: format ## Alias for format

# ─────────────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build Docker image
	@printf "$(BLUE)Building Docker image...$(RESET)\n"
	docker build -t $(IMAGE) .
	@printf "$(GREEN)✓ Image built: $(IMAGE)$(RESET)\n"

.PHONY: docker-run
docker-run: docker-build ## Run Docker container with GPU
	@printf "$(BLUE)Running Docker container...$(RESET)\n"
	docker run --rm --gpus all -p 8080:8080 \
		-v $(MODELS_DIR):/models:ro $(IMAGE)

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts and caches
	@printf "$(BLUE)Cleaning up...$(RESET)\n"
	@rm -rf $(VENV_DIR)
	@rm -rf .pytest_cache .ruff_cache .mypy_cache
	@rm -rf build dist *.egg-info
	@rm -rf htmlcov .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@printf "$(GREEN)✓ Cleanup complete$(RESET)\n"

.PHONY: clean-all
clean-all: clean ## Remove everything including models
	@printf "$(BLUE)Deep cleaning...$(RESET)\n"
	@rm -rf $(EXT_DEPS_DIR)
	@rm -rf $(MODELS_DIR)
	@printf "$(GREEN)✓ Deep clean complete$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: version
version: ## Show version information
	@printf "$(BOLD)Avatar Renderer MCP$(RESET)\n"
	@printf "Version: $(GREEN)0.1.0$(RESET)\n"
	@printf "Author:  Ruslan Magana Vsevolodovna\n"
	@printf "Website: https://ruslanmv.com\n"

.PHONY: check
check: lint test ## Run all checks (lint + test)
	@printf "$(GREEN)✓ All checks passed$(RESET)\n"
