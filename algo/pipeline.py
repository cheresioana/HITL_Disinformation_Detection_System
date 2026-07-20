"""
Narrative tree training and evaluation pipeline.

Shared functions for training dual narrative trees and evaluating them
across multiple thresholds. Used by all entry points in commands/.
"""

# Standard library
import glob
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import count

# Third-party
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# Local
from algo.create_trees import train_narrative_tree_from_dataframe
from algo.algo_utils import load_structure_file
from algo.get_label_dual import get_label_dual_v3
from LLM.orchestrator import fetch_embedding, fetch_embeddings
from utils import clean_text
from config import EVAL_WORKERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Tree discovery
# ---------------------------------------------------------------------------

def discover_structure_files(folder):
    """Find all full_result_*.json files in folder/results/ and return sorted by threshold."""
    results_dir = os.path.join(folder, "results") if folder else "results"
    pattern = os.path.join(results_dir, "full_result_*.json")
    files = glob.glob(pattern)

    parsed = []
    for f in files:
        basename = os.path.basename(f)
        m = re.search(r'full_result_(.+?)\.json', basename)
        if m:
            try:
                threshold = float(m.group(1))
            except ValueError:
                threshold = 0.0
            parsed.append((threshold, f))

    parsed.sort(key=lambda x: x[0])
    return parsed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_narrative_tree(dataset_name, train_df, folder=""):
    """Train narrative trees from a training dataframe."""
    logger.info("=" * 70)
    logger.info("TRAINING - Narrative Tree - %s", dataset_name)
    logger.info("=" * 70)
    logger.info("Training on %d samples...", len(train_df))

    start_time = time.time()
    try:
        train_narrative_tree_from_dataframe(train_df, folder=folder)
        training_time = time.time() - start_time
        logger.info("Training completed in %.2f minutes", training_time / 60)
        return training_time
    except Exception as e:
        logger.error("Training failed: %s", e)
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def precompute_embeddings(df):
    """Compute embeddings for all rows once. Returns a list aligned with df index."""
    logger.info("Pre-computing embeddings for %d samples...", len(df))
    embeddings = fetch_embeddings([clean_text(t) for t in df['text']])
    logger.info("Embeddings computed.")
    return embeddings


def evaluate_dual_on_dataset(df, fake_matches, real_matches, split_name,
                             embeddings=None):
    """Evaluate dual-tree classification on a dataset.

    If *embeddings* is provided, use them instead of recomputing.
    """
    n = len(df)
    logger.info("Evaluating dual on %s (%d samples, %d workers)...",
                split_name, n, EVAL_WORKERS)

    # Results land by index so order matches df regardless of completion order.
    predictions = [0] * n
    rows = list(enumerate(df.iterrows()))
    done = count(1)

    def _judge_one(item):
        i, (idx, row) = item
        text = row['text']
        row_data = {
            'text': text,
            'embedding': embeddings[i] if embeddings is not None
                         else fetch_embedding(clean_text(text)),
        }

        try:
            label, narrative, f_score, r_score, _ = get_label_dual_v3(
                row_data, fake_matches, real_matches, real_label=row['label']
            )
        except Exception as e:
            logger.error("Error at %d: %s", i, e)
            label = 0

        finished = next(done)
        if finished % 50 == 0:
            logger.info("  Progress: %d/%d", finished, n)
        return i, label

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        for i, label in executor.map(_judge_one, rows):
            predictions[i] = label

    return evaluate_model(df['label'].values, np.array(predictions))


def run_dual_comparative_evaluation(dataset_name, val, test, folder=""):
    """Load ALL matching fake/real structure files, evaluate dual-tree on val."""
    logger.info("=" * 70)
    logger.info("DUAL-TREE COMPARATIVE EVALUATION - %s", dataset_name)
    logger.info("=" * 70)

    fake_files = discover_structure_files(folder + "false/")
    real_files = discover_structure_files(folder + "true/")

    if not fake_files:
        logger.warning("No fake tree files in '%sfalse/results/'", folder)
        return None
    if not real_files:
        logger.warning("No real tree files in '%strue/results/'", folder)
        return None

    logger.info("Fake trees: %d (thresholds: %s -> %s)",
                len(fake_files), fake_files[0][0], fake_files[-1][0])
    logger.info("Real trees: %d (thresholds: %s -> %s)",
                len(real_files), real_files[0][0], real_files[-1][0])

    all_results = []
    real_by_threshold = {t: path for t, path in real_files}

    # Pre-compute embeddings once (reused across all thresholds)
    val_embeddings = precompute_embeddings(val)

    for fake_t, fake_path in fake_files:
        if fake_t not in real_by_threshold:
            logger.info("Skipping threshold=%.2f (no matching real tree)", fake_t)
            continue
        real_path = real_by_threshold[fake_t]

        logger.info("-" * 60)
        logger.info("Threshold=%.2f", fake_t)

        try:
            _, fake_matches = load_structure_file(fake_path)
            _, real_matches = load_structure_file(real_path)
            logger.info("Fake nodes: %d, Real nodes: %d",
                        len(fake_matches), len(real_matches))
        except Exception as e:
            logger.error("Failed to load: %s", e)
            continue

        start_time = time.time()
        val_metrics = evaluate_dual_on_dataset(
            val, fake_matches, real_matches, f"val (t={fake_t})",
            embeddings=val_embeddings,
        )
        eval_time = time.time() - start_time

        logger.info("Val  -> Acc: %.4f  P: %.4f  R: %.4f  F1: %.4f  CM: %s",
                    val_metrics['accuracy'], val_metrics['precision'],
                    val_metrics['recall'], val_metrics['f1'],
                    val_metrics['confusion_matrix'])
        logger.info("Eval time: %.1fs", eval_time)

        all_results.append({
            'threshold': fake_t,
            'val': val_metrics,
            'eval_time_s': eval_time,
        })

    if not all_results:
        logger.warning("No results to compare.")
        return None

    # Summary
    logger.info("=" * 100)
    logger.info("COMPARATIVE RESULTS TABLE - DUAL TREE - %s", dataset_name)
    logger.info("=" * 100)

    for r in all_results:
        cm = r['val']['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        logger.info("t=%.2f | Val Acc: %.4f | Val F1: %.4f | TP: %d  TN: %d  FP: %d  FN: %d",
                    r['threshold'], r['val']['accuracy'], r['val']['f1'],
                    tp, tn, fp, fn)

    best = max(all_results, key=lambda r: r['val']['f1'])
    logger.info("Best by Val F1: t=%.2f (F1=%.4f, Acc=%.4f)",
                best['threshold'], best['val']['f1'], best['val']['accuracy'])

    rows = []
    for r in all_results:
        cm = r['val']['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        rows.append({
            'threshold': r['threshold'],
            'val_accuracy': r['val']['accuracy'],
            'val_precision': r['val']['precision'],
            'val_recall': r['val']['recall'],
            'val_f1': r['val']['f1'],
            'val_TP': tp, 'val_TN': tn, 'val_FP': fp, 'val_FN': fn,
        })
    csv_path = os.path.join(folder or ".", "dual_comparative_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    return all_results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(dataset_name, loader_fn, folder, skip_training=False):
    """Full pipeline: train narrative tree + evaluate all thresholds."""
    logger.info("#" * 70)
    logger.info("# NARRATIVE TREE - %s", dataset_name)
    logger.info("#" * 70)

    train, val, test = loader_fn(include_test=True)
    #train = train.head(10)
    #val = val.head(10)
    #test = test.head(10)

    # Step 1: Train
    training_time = None
    if not skip_training:
        training_time = train_narrative_tree(dataset_name, train, folder=folder)
        if training_time is None:
            logger.warning("Skipping evaluation for %s (training failed)", dataset_name)
            return None

    # Step 2: Evaluate all structure files comparatively
    results = run_dual_comparative_evaluation(
        dataset_name, val, test, folder=folder
    )
    return {
        'dataset': dataset_name,
        'training_time': training_time,
        'results': results,
    }
