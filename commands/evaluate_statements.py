"""
Evaluate the narrative algorithm on a validation dataset.

Sweeps over tree-construction thresholds, loads dual (fake + real) trees
for each threshold, classifies every row with get_label_dual_v3, and
reports accuracy / confusion matrix / precision / recall / F1.

Usage:
    python -m algo.eval --file data/work_data/val.csv \
                        --fake-dir narrative_mbd/false/results \
                        --real-dir narrative_mbd/true/results
"""

import argparse
import ast
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from algo.algo_utils import load_structure_file
from algo.get_label_dual import get_label_dual_v3, set_verbose

logger = logging.getLogger(__name__)


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_threshold(df, fake_matches, real_matches):
    """Run get_label_dual_v3 on every row.

    Returns (predictions, rows) where ``rows`` is a list of per-example dicts
    suitable for dumping to CSV (one record per validation row).
    """
    predictions = []
    rows = []
    total = len(df)
    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        label, node, fake_score, real_score, explanation = get_label_dual_v3(
            row, fake_matches, real_matches, real_label=row.get('label', 'unknown'),
        )
        predictions.append(label)
        rate = i / max(time.time() - t_start, 1e-6)
        eta = (total - i) / rate if rate > 0 else 0
        print(f">>> PROGRESS {i}/{total}  ({rate:.2f} rows/s, ETA {eta:.0f}s)", flush=True)
        rows.append({
            "text": row.get("text"),
            "gold": int(row["label"]) if "label" in row else None,
            "pred": int(label),
            "fake_score": float(fake_score) if fake_score is not None else None,
            "real_score": float(real_score) if real_score is not None else None,
            "fake_match": explanation.get("fake_match_text") if isinstance(explanation, dict) else None,
            "real_match": explanation.get("real_match_text") if isinstance(explanation, dict) else None,
            "mode": explanation.get("mode") if isinstance(explanation, dict) else None,
            "narrative": node.text if node is not None else None,
        })
    return predictions, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate narrative algorithm on a validation dataset"
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to the validation CSV (must have 'text', 'embedding', 'label')",
    )
    parser.add_argument(
        "--fake-dir", default="results/narrative_mbd_new/false/results",
        help="Directory containing fake-tree JSONs (default: narrative_mbd/false/results)",
    )
    parser.add_argument(
        "--real-dir", default="results/narrative_mbd_new/true/results",
        help="Directory containing real-tree JSONs (default: narrative_mbd/true/results)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Max samples per label (default: 50, 0 = all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query [FAKE]/[REAL]/[decision] traces from get_label_dual_v3",
    )
    parser.add_argument(
        "--save-predictions",
        help="If set, dump per-row predictions to this CSV path "
             "(text, gold, pred, fake_score, real_score, fake_match, real_match, mode).",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=None,
        help="Tree thresholds to evaluate, e.g. --thresholds 0.55 "
             "(default: sweep 0.6).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    if args.verbose:
        set_verbose(True)

    df = pd.read_csv(args.file)
    df['label'] = df['label'].apply(lambda x: 1 if not x else 0)
    logger.info("Loaded %s  shape=%s  label_counts=%s",
                args.file, df.shape, df['label'].value_counts().to_dict())

    if args.sample > 0:
        df = (
            df.groupby('label', group_keys=False)
            .apply(lambda g: g.head(args.sample))
            .reset_index(drop=True)
        )
        logger.info("Sampled %d per label -> %d rows", args.sample, len(df))

    if 'embedding' not in df.columns:
        logger.info("No 'embedding' column — computing embeddings for %d rows", len(df))
        from LLM.orchestrator import fetch_embeddings
        from utils import clean_text
        df['embedding'] = fetch_embeddings([clean_text(str(t)) for t in df['text']])
    else:
        df['embedding'] = df['embedding'].apply(ast.literal_eval)

    best_acc = 0.0
    best_threshold = None

    thresholds = args.thresholds if args.thresholds else list(np.arange(0.5, 0.55, 0.05))
    for threshold in thresholds:
        threshold = round(threshold, 2)
        fake_path = f"{args.fake_dir}/full_result_{threshold}.json"
        real_path = f"{args.real_dir}/full_result_{threshold}.json"

        logger.info("Threshold %.2f — loading trees: %s, %s",
                     threshold, fake_path, real_path)

        _, fake_matches = load_structure_file(fake_path)
        _, real_matches = load_structure_file(real_path)
        logger.info("  fake nodes: %d  real nodes: %d",
                     len(fake_matches), len(real_matches))

        predictions, per_row = evaluate_threshold(df, fake_matches, real_matches)
        true_labels = df['label'].tolist()

        if args.save_predictions:
            out_path = (
                args.save_predictions
                if len(np.arange(0.5, 0.55, 0.05)) == 1
                else f"{args.save_predictions.rsplit('.', 1)[0]}_t{threshold}.csv"
            )
            pd.DataFrame(per_row).to_csv(out_path, index=False)
            logger.info("  Per-row predictions written to %s", out_path)

        acc = accuracy_score(true_labels, predictions)
        # Weighted averaging to match make train (algo/pipeline.py evaluate_model),
        # so eval-val and train report directly comparable F1/precision/recall.
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

        logger.info("  Accuracy:  %.4f", acc)
        logger.info("  Precision: %.4f  Recall: %.4f  F1: %.4f", prec, rec, f1)
        logger.info("  Confusion matrix:\n%s", cm)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    logger.info("Best threshold: %.2f  (accuracy %.4f)", best_threshold, best_acc)
