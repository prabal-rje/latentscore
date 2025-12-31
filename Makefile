.PHONY: install format fmt-check lint typecheck test check run

install:
	pip install -r requirements.txt

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
