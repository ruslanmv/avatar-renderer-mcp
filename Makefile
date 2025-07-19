# ──────────────────────────────────────────────────────────────────────────────
# Avatar‑Renderer‑Pod  –  Developer Makefile
#
#   make help          # list all targets
#   make setup         # create .venv + install deps
#   make download-models
#   make run           # run FastAPI (localhost:8080)
#   make run-stdio     # run MCP stdio server
#   make test          # run pytest (+CPU stub)
#   make lint / fmt    # flake8 / black
#   make docker-build / docker-run
#   make clean         # wipe .venv, __pycache__, dist
# ──────────────────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------- Globals --
PY          ?= python3.11
VENV_DIR    ?= .venv
IMAGE       ?= avatar-renderer:dev
MODELS_DIR  ?= $(CURDIR)/models
GREEN       := \033[0;32m
RESET       := \033[0m

define PRINT_TARGET
	@printf "$(GREEN)%-18s$(RESET) %s\n" "$(1)" "$(2)"
endef

# --------------------------------------------------------------------- Help ---
.PHONY: help
help:            ## Show this help
	@echo "Avatar‑Renderer‑Pod — developer workflow"
	@echo
	$(call PRINT_TARGET,setup,Create Python venv & install requirements)
	$(call PRINT_TARGET,download-models,Fetch FOMM / Diff2Lip / GFPGAN weights)
	$(call PRINT_TARGET,run,Run FastAPI   (REST /render) on :8080)
	$(call PRINT_TARGET,run-stdio,Run MCP stdio server (render_avatar))
	$(call PRINT_TARGET,lint,Run flake8 lint)
	$(call PRINT_TARGET,fmt,Auto‑format via Black)
	$(call PRINT_TARGET,test,Run pytest suite (CPU stubs))
	$(call PRINT_TARGET,docker-build,Build CUDA image $(IMAGE))
	$(call PRINT_TARGET,docker-run,Run image with GPU + model mount)
	$(call PRINT_TARGET,clean,Remove venv, build artefacts)
	@echo

# ----------------------------------------------------------- Python tooling --
.PHONY: setup
setup: $(VENV_DIR)/bin/activate          ## Create venv + pip install
$(VENV_DIR)/bin/activate:
	@echo "🔧  Creating venv → $(VENV_DIR)"
	$(PY) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip wheel
	$(VENV_DIR)/bin/pip install -r requirements.txt \
		"black==24.4.*" "flake8==7.*" "pytest"
	@echo "✅  venv ready.  Use:  source $(VENV_DIR)/bin/activate"

# ------------------------------------------------------------ Model assets --
.PHONY: download-models
download-models: ## Pull all model checkpoints into ./models
	@bash scripts/download_models.sh "$(MODELS_DIR)"

# ---------------------------------------------------------- Dev run targets --
.PHONY: run
run: setup                               ## Start FastAPI REST server on :8080
	@source $(VENV_DIR)/bin/activate && \
		uvicorn app.api:app --host 0.0.0.0 --port 8080 --reload

.PHONY: run-stdio
run-stdio: setup                         ## Start MCP stdio server
	@source $(VENV_DIR)/bin/activate && \
		python app/mcp_server.py

# -------------------------------------------------------------- Quality CI --
.PHONY: lint
lint: setup                              ## flake8 lint
	@$(VENV_DIR)/bin/flake8 app tests

.PHONY: fmt
fmt: setup                               ## Black code format
	@$(VENV_DIR)/bin/black app tests

.PHONY: test
test: setup                              ## Run pytest (CPU stub)
	@$(VENV_DIR)/bin/pytest -q

# ---------------------------------------------------------------- Docker ---
.PHONY: docker-build
docker-build:                            ## Build CUDA image (avatar-renderer:dev)
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run: docker-build                 ## Run container with GPU & models mount
	docker run --rm --gpus all -p 8080:8080 \
	  -v $(MODELS_DIR):/models:ro $(IMAGE)

# ---------------------------------------------------------------- Cleanup --
.PHONY: clean
clean:                                   ## Delete venv + pycache
	rm -rf $(VENV_DIR) **/__pycache__ dist build .pytest_cache
	@echo "🧹  Clean complete."
