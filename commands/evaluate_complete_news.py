"""Evaluate on complete news articles."""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Path setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import COMPLETE_NEWS_TEST, NARRATIVE_DIR
from algo.algo_utils import load_structure_file
from algo.parse_news import worker

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate on complete news articles")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Tree threshold to load (default 0.3)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers (default 5)")
    args = parser.parse_args()

    # ── Load dataset ──
    csv_path = COMPLETE_NEWS_TEST
    df = pd.read_csv(csv_path, encoding='utf-8')

    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")

    df['label'] = df['label'].map({True: 0, False: 1, 'True': 0, 'False': 1})
    if df['label'].isna().any() or not set(df['label'].unique()).issubset({0, 1}):
        raise ValueError("label column must contain only True/False with no NaNs")

    logger.info("Loaded %d news articles (fake=%d, real=%d)",
                len(df), (df['label'] == 1).sum(), (df['label'] == 0).sum())

    # ── Load dual trees ──
    t = args.threshold
    fake_path = str(NARRATIVE_DIR / "false" / "results" / f"full_result_{t}.json")
    real_path = str(NARRATIVE_DIR / "true" / "results" / f"full_result_{t}.json")
    logger.info("Loading trees: fake=%s, real=%s", fake_path, real_path)

    _, fake_matches = load_structure_file(fake_path)
    _, real_matches = load_structure_file(real_path)
    logger.info("Tree sizes: fake=%d nodes, real=%d nodes",
                len(fake_matches), len(real_matches))

    # ── Parallel evaluation ──
    predicted_label = [None] * len(df)
    true_label = df['label'].tolist()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(worker, i, r['title'], r['summary'],
                      fake_matches, real_matches, r['label'])
            for i, r in df.iterrows()
        ]
        for fut in as_completed(futures):
            i, lab = fut.result()
            predicted_label[i] = lab

    # ── Results ──
    true_label = np.array(true_label, dtype=int)
    predicted_label = np.array(predicted_label, dtype=int)

    acc = accuracy_score(true_label, predicted_label)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_label, predicted_label, average='binary', pos_label=1,
    )

    logger.info("Evaluation results:")
    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall:    %.4f", rec)
    logger.info("  F1-score:  %.4f", f1)

    cm = confusion_matrix(true_label, predicted_label, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    logger.info("Confusion matrix:\n%s", cm_df)
