"""Incrementally update narrative trees with new labeled data."""

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

import pandas as pd

from algo.update_model import update_trees_with_new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Incrementally update narrative trees with new labeled data'
    )
    parser.add_argument('--data', required=True,
                        help='CSV file with new labeled data (columns: text, label)')
    parser.add_argument('--folder', default='narrative_mbd/',
                        help='Tree folder prefix (default: narrative_mbd/)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Cross-encoder match threshold for grafting (default: 1.0)')
    args = parser.parse_args()

    new_df = pd.read_csv(args.data)
    update_trees_with_new_data(new_df, folder=args.folder, match_threshold=args.threshold)
