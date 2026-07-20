"""Run SOTA baseline algorithms (SVM, LR, GradientBoosting, DecisionTree, KNN) on all datasets."""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'sota_comparison' / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)

from sota_comparison.run_all_experiments import main

if __name__ == '__main__':
    main()
