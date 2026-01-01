.PHONY: install setup format fmt-check lint typecheck test check run download-model

# --- GUARD CLAUSE ---
ifndef ENV_NAME
$(error ❌ Error: ENV_NAME is missing. Usage: make install ENV_NAME=latentscore)
endif

# --- CONFIG ---
MODELS_DIR := models
LLM_DIR := $(MODELS_DIR)/gemma-3-270m-it-qat-4bit
LLM_REPO := mlx-community/gemma-3-270m-it-qat-4bit

# --- SYSTEM CHECKS ---
check-system:
	@echo "🔍 Checking System Dependencies..."
	@which sox >/dev/null 2>&1 || (echo "❌ SoX missing. Run: brew install sox" && exit 1)
	@echo "✅ System binaries look good."

# --- THE JUICED INSTALLER ---
install: check-system
	@echo "🍏 Setting up M4-Optimized Environment: $(ENV_NAME)..."
	conda env update --name $(ENV_NAME) --file environment.yml --prune
	@echo "✅ Install complete! Don't forget to run: conda activate $(ENV_NAME)"

# Alias
setup: install

# --- MODEL FETCHING ---
download-model:
	@echo "📥 Downloading Gemma 3 270M QAT (MLX)..."
	@mkdir -p $(LLM_DIR)
	@python -c "from huggingface_hub import snapshot_download; snapshot_download('$(LLM_REPO)', local_dir='$(LLM_DIR)')"
	@echo "✅ Model ready in $(LLM_DIR)/"

# --- DEV TOOLS ---
format:
	ruff format .

fmt-check:
	ruff format --check .

lint:
	ruff check .

typecheck:
	pyright

test:
	pytest

check: lint fmt-check typecheck test

run:
	python -m app