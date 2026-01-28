.PHONY: install dev-install setup format fmt-check lint typecheck test check run download-models download-llm download-embeddings

# 1. Configuration
ENV_NAME ?= latentscore
MODELS_DIR := models
LLM_DIR := $(MODELS_DIR)/latentscore-gemma3-270m-v5-merged
LLM_REPO := guprab/latentscore-gemma3-270m-v5-merged
EMBED_DIR := $(MODELS_DIR)/all-MiniLM-L6-v2
EMBED_REPO := sentence-transformers/all-MiniLM-L6-v2

# 2. Environment Detection (The Magic Fix)
# Check if 'conda' command exists. If yes, use it. If no, use standard python.
CONDA_EXE := $(shell command -v conda 2> /dev/null)

ifdef CONDA_EXE
    # We are on Local Mac (Conda)
    RUN_CMD := conda run -n $(ENV_NAME) --no-capture-output
    PIP_CMD := $(RUN_CMD) pip
    PYTHON_CMD := $(RUN_CMD) python
    MSG_PREFIX := "🍏 (Conda)"
else
    # We are on Remote Linux (Standard Python/Venv)
    RUN_CMD := 
    PIP_CMD := pip
    PYTHON_CMD := python
    MSG_PREFIX := "🐧 (Standard)"
endif

check-system:
	@echo "🔍 Checking system..."
	@which sox >/dev/null 2>&1 || (echo "❌ SoX missing. Run: brew install sox (Mac) or apt install sox (Linux)" && exit 1)
	@if [ "$$(uname)" = "Linux" ]; then \
		test -f /usr/include/alsa/asoundlib.h || (echo "❌ ALSA headers missing (Linux only). Run: apt install libasound2-dev" && exit 1); \
	fi

install: check-system
	@echo $(MSG_PREFIX) "Installing dependencies..."
ifdef CONDA_EXE
	# Create Conda env if missing
	conda create -n $(ENV_NAME) python=3.10 -y || true
	# Install pip-tools inside Conda
	$(PIP_CMD) install pip-tools
else
	# Ensure pip-tools is installed in current venv
	$(PIP_CMD) install pip-tools
endif
	# Sync requirements (Handles the 'mlx' condition automatically!)
	$(RUN_CMD) pip-sync data_work/requirements.txt
	@echo "✅ Dependencies synced."

dev-install:
	@echo $(MSG_PREFIX) "Installing Dev Tools..."
	$(PIP_CMD) install ruff pyright pytest
	@echo "✅ Dev tools installed."

setup: install dev-install download-models

# 3. Model Management
download-models: download-llm download-embeddings

download-llm:
	@echo "📥 Downloading LLM..."
	@mkdir -p $(LLM_DIR)
	$(PYTHON_CMD) -c "from huggingface_hub import snapshot_download; snapshot_download('$(LLM_REPO)', local_dir='$(LLM_DIR)')"

download-embeddings:
	@echo "📥 Downloading embeddings..."
	@mkdir -p $(EMBED_DIR)
	$(PYTHON_CMD) -c "from huggingface_hub import snapshot_download; snapshot_download('$(EMBED_REPO)', local_dir='$(EMBED_DIR)')"

# 4. Code Quality & Run
# RUN_CMD automatically adapts to 'conda run' or '' (empty)

format:
	$(RUN_CMD) ruff format .

fmt-check:
	$(RUN_CMD) ruff format --check .

lint:
	$(RUN_CMD) ruff check .

typecheck:
	$(RUN_CMD) pyright

test:
	$(RUN_CMD) pytest

check: lint fmt-check typecheck test

run:
	$(PYTHON_CMD) -m app
