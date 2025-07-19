# ──────────────────────────────────────────────────────────────────────────────
# Avatar‑Renderer‑Pod  –  Developer Makefile
#
#   make help           # list all targets
#   make setup          # create .venv + install all dependencies + models
#   make install-git-deps  # clone & install non-package Git repos
#   make download-models # fetch all model checkpoints
#   make run            # run FastAPI (localhost:8080)
#   make run-stdio      # run MCP stdio server
#   make test           # run pytest (+CPU stub)
#   make lint / fmt     # flake8 / black
#   make docker-build / docker-run
#   make clean          # wipe .venv, **pycache**, dist
# ──────────────────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------- Globals --
PY             ?= python3.11
VENV_DIR       ?= .venv
VENV_BIN       = $(VENV_DIR)/bin
EXT_DEPS_DIR   ?= external_deps
IMAGE          ?= avatar-renderer:dev
MODELS_DIR     ?= $(CURDIR)/models
GREEN          := \033[0;32m
RESET          := \033[0m

# Helper for printing colored target help text
define PRINT_TARGET
	@printf "$(GREEN)%-20s$(RESET) %s\n" "$(1)" "$(2)"
endef

# --------------------------------------------------------------------- Help ---
.PHONY: help
help:           ## Show this help
	@echo "Avatar‑Renderer‑Pod — developer workflow"
	@echo
	$(call PRINT_TARGET,setup,Create venv, install deps & models)
	$(call PRINT_TARGET,install-git-deps,Clone & install non-package Git repos)
	$(call PRINT_TARGET,download-models,Fetch FOMM / Diff2Lip / GFPGAN weights)
	$(call PRINT_TARGET,run,Run FastAPI (REST) on :8080)
	$(call PRINT_TARGET,run-stdio,Run MCP stdio server)
	$(call PRINT_TARGET,lint,Run flake8 lint)
	$(call PRINT_TARGET,fmt,Auto-format via Black)
	$(call PRINT_TARGET,test,Run pytest suite)
	$(call PRINT_TARGET,docker-build,Build CUDA image $(IMAGE))
	$(call PRINT_TARGET,docker-run,Run image with GPU & models mount)
	$(call PRINT_TARGET,clean,Remove venv, build artefacts)
	@echo

# ----------------------------------------------------------- Python tooling --
# The main setup target: venv + packages + Git deps + model assets
.PHONY: setup
setup: $(VENV_BIN)/activate    ## Create venv + install all dependencies & models

$(VENV_BIN)/activate: pyproject.toml
	@echo "🔧   Creating venv → $(VENV_DIR)"
	$(PY) -m venv $(VENV_DIR)
	@echo "🐍   Installing standard dependencies from pyproject.toml..."
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[dev]"
	@$(MAKE) install-git-deps
	@$(MAKE) download-models
	@echo "🔗   Symlinking models → /models"
	@ln -sf $(MODELS_DIR) /models
	@echo "✅   venv & models ready. To activate: source $(VENV_BIN)/activate"
	@echo "     External Git dependencies are in: $(EXT_DEPS_DIR)/"

# ---------------------------------------------------------------- Git deps --
.PHONY: install-git-deps
install-git-deps:           ## Clone and install non-package Git repos
	@echo "📂   Cloning and installing non-package Git dependencies..."
	@mkdir -p $(EXT_DEPS_DIR)
	# SadTalker
	@if [ ! -d "$(EXT_DEPS_DIR)/SadTalker" ]; then \
		echo "Cloning SadTalker..."; \
		git clone https://github.com/OpenTalker/SadTalker.git $(EXT_DEPS_DIR)/SadTalker; \
	fi
	$(VENV_BIN)/pip install -r $(EXT_DEPS_DIR)/SadTalker/requirements.txt
	# first-order-model
	@if [ ! -d "$(EXT_DEPS_DIR)/first-order-model" ]; then \
		echo "Cloning first-order-model..."; \
		git clone https://github.com/AliaksandrSiarohin/first-order-model.git $(EXT_DEPS_DIR)/first-order-model; \
	fi
	# Wav2Lip
	@if [ ! -d "$(EXT_DEPS_DIR)/Wav2Lip" ]; then \
		echo "Cloning Wav2Lip..."; \
		git clone https://github.com/Rudrabha/Wav2Lip.git $(EXT_DEPS_DIR)/Wav2Lip; \
	fi
	$(VENV_BIN)/pip install -r $(EXT_DEPS_DIR)/Wav2Lip/requirements.txt
	@echo "✅   Git dependencies installed."

# ------------------------------------------------------------ Model assets --
.PHONY: download-models
download-models:             ## Pull all model checkpoints into ./models
	@echo "🔽   Downloading model checkpoints into $(MODELS_DIR)"
	@bash -lc "source $(VENV_BIN)/activate && \
	             bash scripts/download_models.sh '$(MODELS_DIR)'" || true
	@echo "✅   Model checkpoints downloaded."

# ---------------------------------------------------------- Dev run targets --
run: setup                   ## Start FastAPI REST server on :8080
	@echo "🚀 Starting FastAPI server on http://localhost:8080"
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/uvicorn app.api:app --host 0.0.0.0 --port 8080 --reload

run-stdio: setup             ## Start MCP stdio server
	@echo "🚀 Starting MCP stdio server..."
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/python app/mcp_server.py

# -------------------------------------------------------------- Quality CI --
lint: setup                  ## flake8 lint
	@$(VENV_BIN)/flake8 app tests

fmt: setup                   ## Black code format
	@$(VENV_BIN)/black app tests

test: setup                  ## Run pytest (CPU stub)
	@PYTHONPATH=$(CURDIR)/$(EXT_DEPS_DIR):$(PYTHONPATH) \
		$(VENV_BIN)/pytest -q

# ---------------------------------------------------------------- Docker ---
docker-build:                ## Build CUDA image (avatar-renderer:dev)
	docker build -t $(IMAGE) .

docker-run: docker-build     ## Run container with GPU & models mount
	docker run --rm --gpus all -p 8080:8080 \
		-v $(MODELS_DIR):/models:ro $(IMAGE)

# ---------------------------------------------------------------- Cleanup --
.PHONY: clean
clean:                       ## Delete venv, pycache, and cloned repos
	@echo "🧹   Cleaning up project..."
	rm -rf $(VENV_DIR) dist build .pytest_cache .ruff_cache $(EXT_DEPS_DIR)
	# FIX: Use find to remove all __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "🧹   Clean complete."
