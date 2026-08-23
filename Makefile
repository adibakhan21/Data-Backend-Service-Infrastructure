.PHONY: help install data api dashboard test bench eval agents eval-agents scale docker clean

help:
	@echo "install    - create .venv and install dependencies"
	@echo "data       - generate 200k synthetic trades into SQLite"
	@echo "api        - run the FastAPI server on :8000"
	@echo "dashboard  - run the Streamlit dashboard on :8501"
	@echo "test       - run the pytest suite"
	@echo "bench      - run benchmarks, write benchmarks/results.md"
	@echo "eval       - score the anomaly detector, write benchmarks/detector_eval.md"
	@echo "agents     - triage anomalies through the LangGraph agent (no key needed)"
	@echo "eval-agents- score the agent: tool selection, output validity, routing"
	@echo "scale      - profile snapshot build + memory as the dataset grows"
	@echo "docker     - build and start api + dashboard via docker compose"
	@echo "clean      - remove the database, caches and generated artefacts"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python scripts/generate_data.py --rows 200000 --reset

api:
	.venv/bin/uvicorn tradetrack.api.main:app --reload --app-dir src

dashboard:
	.venv/bin/streamlit run dashboard/app.py

test:
	.venv/bin/python -m pytest

bench:
	.venv/bin/python scripts/run_benchmarks.py

eval:
	.venv/bin/python scripts/evaluate_detector.py

agents:
	.venv/bin/python scripts/triage_demo.py

eval-agents:
	.venv/bin/python scripts/evaluate_agents.py --backend auto

scale:
	.venv/bin/python scripts/scale_profile.py

docker:
	docker compose --profile seed run --rm seed
	docker compose up --build

clean:
	rm -rf data/*.db data/*.db-wal data/*.db-shm .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
