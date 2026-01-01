.PHONY: install setup format fmt-check lint typecheck test check run download-models download-llm download-embeddings

ifndef ENV_NAME
$(error ❌ ENV_NAME missing. Usage: make install ENV_NAME=latentscore)
endif

MODELS_DIR := models
LLM_DIR := $(MODELS_DIR)/gemma-3-1b-it-qat-8bit
LLM_REPO := mlx-community/gemma-3-1b-it-qat-8bit
EMBED_DIR := $(MODELS_DIR)/all-MiniLM-L6-v2
EMBED_REPO := sentence-transformers/all-MiniLM-L6-v2

check-system:
	@echo "🔍 Checking system..."
	@which sox >/dev/null 2>&1 || (echo "❌ SoX missing. Run: brew install sox" && exit 1)
	@echo "✅ System OK"

install: check-system
	@echo "🍏 Setting up: $(ENV_NAME)..."
	conda env update --name $(ENV_NAME) --file environment.yml --prune
	@echo "✅ Done! Run: conda activate $(ENV_NAME)"

setup: install

download-models: download-llm download-embeddings
	@echo "✅ All models ready"

download-llm:
	@echo "📥 Downloading Gemma 3 1B IT (4-bit)..."
	@mkdir -p $(LLM_DIR)
	python -c "from huggingface_hub import snapshot_download; snapshot_download('$(LLM_REPO)', local_dir='$(LLM_DIR)')"
	@echo "✅ LLM ready: $(LLM_DIR)"

download-embeddings:
	@echo "📥 Downloading embeddings..."
	@mkdir -p $(EMBED_DIR)
	python -c "from huggingface_hub import snapshot_download; snapshot_download('$(EMBED_REPO)', local_dir='$(EMBED_DIR)')"
	@echo "✅ Embeddings ready: $(EMBED_DIR)"

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