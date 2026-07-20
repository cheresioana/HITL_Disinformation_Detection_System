"""
Centralized project configuration.

All file paths and directory references live here so they can be
changed in a single place instead of editing multiple modules.
"""

import os
from pathlib import Path

# ── Base directories ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ── Active model threshold ────────────────────────────────────────
ACTIVE_THRESHOLD = 0.5

# ── Narrative tree paths (derived from threshold) ─────────────────
NARRATIVE_DIR = RESULTS_DIR / "narrative_mbd_new"
FAKE_TREE_PATH = str(
    NARRATIVE_DIR / "false" / "results" / f"full_result_{ACTIVE_THRESHOLD}.json"
)
REAL_TREE_PATH = str(
    NARRATIVE_DIR / "true" / "results" / f"full_result_{ACTIVE_THRESHOLD}.json"
)

# ── Romanian variant ──────────────────────────────────────────────
NARRATIVE_DIR_RO = PROJECT_ROOT / "narrative_mbd_ro"

# ── Datasets ──────────────────────────────────────────────────────
TRANSLATIONS_DIR = DATASETS_DIR / "tvr_info" / "json_translations"
COMPLETE_NEWS_TEST = str(
    DATASETS_DIR / "complete_news_data" / "complete_news_test_df.csv"
)
CORRECTION_SET_PATH = (
    DATASETS_DIR / "mindbugs_updated" / "correction_df_prepared.csv"
)

# ── Backup / restore ─────────────────────────────────────────────
BACKUP_DIR = RESULTS_DIR / "backup"
BACKUP_FAKE_TREE_PATH = str(
    BACKUP_DIR / "false" / "results" / f"full_result_{ACTIVE_THRESHOLD}.json"
)
BACKUP_REAL_TREE_PATH = str(
    BACKUP_DIR / "true" / "results" / f"full_result_{ACTIVE_THRESHOLD}.json"
)

# ── Upload settings ──────────────────────────────────────────────
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPDATE_DATASET_PATH = UPLOAD_DIR / "update_dataset.csv"
FULL_DATASET_PATH = UPLOAD_DIR / "full_retrain.csv"
ALLOWED_EXT = {".csv"}
REQUIRED_COLS = {"text", "label"}

# Ollama runs on the RunPod server (has gemma3:12b and gemma3:27b loaded).
# Override the base URL via the OLLAMA_BASE_URL env var when the pod changes
# (RunPod proxy host/id changes on every pod restart).
OLLAMA_MODEL_GEN = "gemma3:27b"
OLLAMA_MODEL_ENT = "gemma3:27b"
OLLAMA_MODEL_SMALL = "gemma3:12b"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    #"http://localhost:11434/",
    "https://sy6pkzmzi4hwqn-11434.proxy.runpod.net",
)
OLLAMA_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

# ── Evaluation parallelism ───────────────────────────────────────
# Number of threads used to run the per-row dual-tree judge during
# evaluation. The judge is I/O-bound on the Ollama HTTP call, so threads
# give near-linear speedup. Match this to the server's OLLAMA_NUM_PARALLEL.
# Override with the EVAL_WORKERS environment variable.
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "16"))
