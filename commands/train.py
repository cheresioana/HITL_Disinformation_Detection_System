"""Train narrative trees on configured datasets."""

import argparse
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

from sota_comparison.src.adapters import (
    load_covid_dataset, load_mindbugs_dataset, load_liar_dataset,
    load_welfake_dataset, load_fakenewsnet_dataset,
)
from algo.pipeline import run_full_pipeline
from config import NARRATIVE_DIR

logger = logging.getLogger(__name__)

# ── Dataset configurations ──
DATASETS = {
    'Mindbugs':     {'loader': load_mindbugs_dataset,    'folder': str(NARRATIVE_DIR) + '/'},
    #'COVID-19':     {'loader': load_covid_dataset,       'folder': 'results/narrative_covid/'},
    #'LIAR':         {'loader': load_liar_dataset,        'folder': 'results/narrative_liar/'},
    #'WELFake':      {'loader': load_welfake_dataset,     'folder': 'results/narrative_welfake/'},
    #'FakeNewsNet':  {'loader': load_fakenewsnet_dataset, 'folder': 'results/narrative_fnn/'},
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train narrative trees.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest checkpoints instead of rebuilding "
             "from scratch (the default overwrites any existing trees).",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("NARRATIVE TREE - TRAIN & EVALUATE ON ALL DATASETS")
    logger.info("=" * 70)
    logger.info("Mode: %s", "RESUME (continue from checkpoints)"
                if args.resume else "FROM SCRATCH (overwrite existing trees)")

    # Set to True to skip training (use existing structure files)
    SKIP_TRAINING = {
        'Mindbugs':    False,
        #'COVID-19':    False,
        #'LIAR':        False,
        #'WELFake':     False,
        #'FakeNewsNet': False,
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
            resume=args.resume,
        )
        if result:
            all_dataset_results[ds_name] = result

    # ── Cross-dataset summary ──
    logger.info("=" * 90)
    logger.info("CROSS-DATASET SUMMARY (best threshold per dataset)")
    logger.info("=" * 90)

    for ds_name, ds_result in all_dataset_results.items():
        results = ds_result['results']
        if not results:
            logger.info("%s: N/A", ds_name)
            continue
        best = max(results, key=lambda r: r['val']['f1'])
        t_min = (f"{ds_result['training_time']/60:.1f}"
                 if ds_result['training_time'] else "cached")
        logger.info("%s: t=%.2f  F1=%.4f  Acc=%.4f  Train=%s min",
                    ds_name, best['threshold'],
                    best['val']['f1'], best['val']['accuracy'], t_min)

    logger.info("All narrative tree experiments completed!")
