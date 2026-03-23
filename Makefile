.PHONY: install download-data train evaluate serve benchmark drift-report lint test

install:
	pip install -e ".[dev]"
	pre-commit install

download-data:
	@echo "CMAPSS data expected at ../CMAPSSData/ (sibling of this repo)."
	@echo "Files needed: train_FD00{1-4}.txt, test_FD00{1-4}.txt, RUL_FD00{1-4}.txt"
	@echo "Download from: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6"

train:
	python -m scripts.train

evaluate:
	python -m scripts.evaluate

serve:
	uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	python -m scripts.benchmark

drift-report:
	python -m scripts.drift

lint:
	black --check .
	ruff check .

test:
	pytest tests/ -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
