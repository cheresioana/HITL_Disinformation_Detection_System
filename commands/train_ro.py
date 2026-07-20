"""Train narrative trees on Romanian datasets."""

import logging
import sys
from pathlib import Path

# Path setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)

# Set language BEFORE any pipeline imports
import constants
constants.LANGUAGE = "RO"

from sota_comparison.src.adapters import load_mindbugs_ro_dataset
from algo.pipeline import run_full_pipeline
from config import NARRATIVE_DIR_RO

logger = logging.getLogger(__name__)

# ── Dataset configuration ──
DATASETS = {
    'Mindbugs-RO': {'loader': load_mindbugs_ro_dataset, 'folder': str(NARRATIVE_DIR_RO) + '/'},
}


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("NARRATIVE TREE (ROMANIAN) - TRAIN & EVALUATE")
    logger.info("=" * 70)
    logger.info("Language: %s", constants.LANGUAGE)

    SKIP_TRAINING = {
        'Mindbugs-RO': True,
    }

    all_dataset_results = {}

    for ds_name, ds_config in DATASETS.items():
        skip = SKIP_TRAINING.get(ds_name, False)
        if skip:
            logger.info("[%s] Skipping training (using existing structures)", ds_name)
        result = run_full_pipeline(
            ds_name,
            ds_config['loader'],
            ds_config['folder'],
            skip_training=skip,
        )
        if result:
            all_dataset_results[ds_name] = result

    # ── Summary ──
    logger.info("=" * 70)
    logger.info("ROMANIAN PIPELINE COMPLETE")
    logger.info("=" * 70)
    for ds_name, ds_result in all_dataset_results.items():
        results = ds_result['results']
        if results:
            best = max(results, key=lambda r: r['val']['f1'])
            logger.info("%s: Best t=%.2f  F1=%.4f  Acc=%.4f",
                        ds_name, best['threshold'],
                        best['val']['f1'], best['val']['accuracy'])
