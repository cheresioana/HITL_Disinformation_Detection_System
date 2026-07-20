"""Quick script to print the size of each dataset (train, val, test)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters import (
    load_covid_dataset, load_mindbugs_dataset, load_liar_dataset,
    load_welfake_dataset, load_fakenewsnet_dataset
)

DATASETS = {
    'COVID-19':    load_covid_dataset,
    'Mindbugs':    load_mindbugs_dataset,
    'LIAR':        load_liar_dataset,
    'WELFake':     load_welfake_dataset,
    'FakeNewsNet': load_fakenewsnet_dataset,
}

header = f"{'Dataset':<15} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7} | {'Fake%':>6}"
print(header)
print("-" * len(header))
if __name__=='__main__':
    for name, loader in DATASETS.items():
        train, val, test = loader(include_test=True)
        total = len(train) + len(val) + len(test)
        fake_pct = (train['label'].sum() + val['label'].sum() + test['label'].sum()) / total * 100
        print(f"{name:<15} | {len(train):>7} | {len(val):>7} | {len(test):>7} | {total:>7} | {fake_pct:>5.1f}%")
