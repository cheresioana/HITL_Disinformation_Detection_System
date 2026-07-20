PYTHON ?= python3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Train narrative trees (English, Mindbugs dataset)
train:
	$(PYTHON) -m commands.train

## Train narrative trees (Romanian, Mindbugs-RO dataset)
train-ro:
	$(PYTHON) -m commands.train_ro

## Incrementally update trees with new data (DATA and FOLDER required)
update:
	$(PYTHON) -m commands.update_model --data $(DATA) --folder $(FOLDER)

## Translate Mindbugs dataset to Romanian
translate-ro:
	$(PYTHON) -m scripts.translate_dataset_ro --concurrency 10

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

## Evaluate on complete news articles
eval-news:
	$(PYTHON) -m commands.evaluate_complete_news --threshold 0.3 --workers 8

## Run SOTA baselines (SVM, LR, GradientBoosting, DecisionTree, KNN) on all datasets
eval-sota:
	$(PYTHON) -m commands.eval_sota

## Print label counts and date distributions for all datasets
dataset-stats:
	$(PYTHON) -m scripts.dataset_stats

## Scan translated JSON articles for fake news
scan-translations:
	$(PYTHON) -m scripts.scan_translations

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
	@echo "  train           Train narrative trees (English, Mindbugs)"
	@echo "  train-ro        Train narrative trees (Romanian, Mindbugs-RO)"
	@echo "  update          Update trees with new data (DATA=path FOLDER=path)"
	@echo "  translate-ro    Translate Mindbugs dataset to Romanian"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval-val        Evaluate on validation dataset (FILE=path required)"
	@echo "  eval-news       Evaluate on complete news articles (600 articles)"
	@echo "  eval-sota       Run SOTA baselines on all datasets"
	@echo "  dataset-stats   Print label counts and date distributions"
	@echo "  scan-translations  Scan translated JSONs for fake news"
	@echo ""
	@echo "Utilities:"
	@echo "  install         Install Python dependencies"
	@echo "  test            Run the full pytest suite (unit + integration)"
	@echo "  test-fast       Run only the fast unit tests (skip integration)"
	@echo "  lint            Syntax-check key Python files"
	@echo "  help            Show this help message"
	@echo ""

.PHONY: train train-ro update translate-ro eval-val eval-news eval-sota dataset-stats scan-translations install test test-fast lint help
.DEFAULT_GOAL := help
