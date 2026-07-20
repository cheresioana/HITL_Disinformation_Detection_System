PYTHON ?= python3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Train narrative trees from scratch (English, Mindbugs; overwrites existing trees)
train:
	$(PYTHON) -m commands.train

## Resume training from the latest checkpoints instead of starting over
train-resume:
	$(PYTHON) -m commands.train --resume

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

## Evaluate on validation dataset (FILE required, optional SAMPLE=N, VERBOSE=1 for per-query traces)
eval-val:
	@if [ -z "$(FILE)" ]; then echo "error: FILE is required, e.g. make eval-val FILE=path/to/val.csv"; exit 2; fi
	@start=$$(date +%s); \
	$(PYTHON) -m commands.evaluate_statements --file $(FILE) $(if $(SAMPLE),--sample $(SAMPLE)) $(if $(VERBOSE),--verbose) $(if $(THRESHOLDS),--thresholds $(THRESHOLDS)); \
	status=$$?; \
	end=$$(date +%s); \
	elapsed=$$((end-start)); \
	printf 'evaluate_statements took %dm %ds\n' $$((elapsed/60)) $$((elapsed%60)); \
	exit $$status

## Evaluate on complete news articles (optional THRESHOLD=N, WORKERS=N)
eval-news:
	$(PYTHON) -m commands.evaluate_complete_news $(if $(THRESHOLD),--threshold $(THRESHOLD)) --workers $(if $(WORKERS),$(WORKERS),8)

## Run SOTA baselines (SVM, LR, GradientBoosting, DecisionTree, KNN) on all datasets
eval-sota:
	$(PYTHON) -m commands.eval_sota

## Print label counts and date distributions for all datasets
dataset-stats:
	$(PYTHON) -m scripts.dataset_stats

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

## Run the Flask API/web app (http://0.0.0.0:5003)
run-api:
	$(PYTHON) app/app.py

## Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

## Run the full pytest suite (unit + integration; integration auto-skips offline)
test:
	$(PYTHON) -m pytest

## Run only the fast unit tests (skip tree/Ollama integration tests)
test-fast:
	$(PYTHON) -m pytest -m "not integration"

## Check syntax of all Python files
lint:
	$(PYTHON) -m py_compile algo/create_trees.py
	$(PYTHON) -m py_compile algo/pipeline.py
	$(PYTHON) -m py_compile algo/algo_utils.py
	$(PYTHON) -m py_compile algo/parse_news.py
	$(PYTHON) -m py_compile algo/get_label_dual.py
	$(PYTHON) -m py_compile commands/train.py
	$(PYTHON) -m py_compile commands/train_ro.py
	$(PYTHON) -m py_compile commands/evaluate_statements.py
	$(PYTHON) -m py_compile commands/evaluate_complete_news.py
	$(PYTHON) -m py_compile commands/eval_sota.py
	$(PYTHON) -m py_compile commands/update_model.py
	$(PYTHON) -m py_compile algo/update_model.py

## Show available commands
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Training:"
	@echo "  train           Train narrative trees from scratch (English, Mindbugs)"
	@echo "  train-resume    Resume training from the latest checkpoints"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval-val        Evaluate on validation dataset (FILE=path required)"
	@echo "  eval-news       Evaluate on complete news articles (600 articles)"
	@echo "  eval-sota       Run SOTA baselines on all datasets"
	@echo "  dataset-stats   Print label counts and date distributions"
	@echo ""
	@echo "Utilities:"
	@echo "  run-api         Run the Flask API/web app (port 5003)"
	@echo "  install         Install Python dependencies"
	@echo "  test            Run the full pytest suite (unit + integration)"
	@echo "  test-fast       Run only the fast unit tests (skip integration)"
	@echo "  lint            Syntax-check key Python files"
	@echo "  help            Show this help message"
	@echo ""

.PHONY: train train-resume eval-val eval-news eval-sota dataset-stats run-api install test test-fast lint help
.DEFAULT_GOAL := help
