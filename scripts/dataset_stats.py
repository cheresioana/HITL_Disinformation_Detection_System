"""Print label counts and date distributions for every dataset in datasets/."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "sota_comparison" / "src"))

import pandas as pd
from sota_comparison.src.adapters import (
    CovidDatasetAdapter,
    MindbugsDatasetAdapter,
    LiarDatasetAdapter,
    FakeNewsNetDatasetAdapter,
)

DATASETS = [
    ("COVID",        CovidDatasetAdapter()),
    ("Mindbugs",     MindbugsDatasetAdapter()),
    ("LIAR",         LiarDatasetAdapter()),
    ("FakeNewsNet",  FakeNewsNetDatasetAdapter()),
]

# Datasets that have a 'date' column in the raw CSV files
DATE_DATASETS = {
    "Mindbugs":    Path("datasets") / "mindbugs",
}


def _split_stats(df, name):
    """Return a formatted stats line for one split."""
    total = len(df)
    real = (df["label"] == 0).sum()
    fake = (df["label"] == 1).sum()
    return f"  {name:<6} {total:>6} total | Real: {real:>5} | Fake: {fake:>5}"


def _print_label_stats(dataset_name, adapter):
    """Load all splits and print label counts."""
    print(f"\n{'=' * 55}")
    print(f"  {dataset_name}")
    print(f"{'=' * 55}")

    train = adapter.load_train()
    print(_split_stats(train, "Train:"))

    val = adapter.load_val()
    print(_split_stats(val, "Val:"))

    try:
        test = adapter.load_test()
        if test is not None:
            print(_split_stats(test, "Test:"))
    except FileNotFoundError:
        pass


def _print_date_distribution(dataset_name, base_path):
    """Load raw CSVs, parse date column, and show label counts per year-month."""
    csv_files = sorted(base_path.glob("*.csv"))
    if not csv_files:
        return

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "date" in df.columns and "label" in df.columns:
                frames.append(df[["date", "label"]])
        except Exception:
            continue

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], format="%d.%m.%Y", errors="coerce")
    combined = combined.dropna(subset=["date"])

    # Normalize label: True/False -> 0/1
    label_map = {"True": 0, "False": 1, True: 0, False: 1}
    if combined["label"].dtype == object or combined["label"].dtype == bool:
        combined["label"] = combined["label"].map(label_map)

    combined["year_month"] = combined["date"].dt.to_period("M")

    pivot = combined.groupby(["year_month", "label"]).size().unstack(fill_value=0)
    pivot.columns = ["Real" if c == 0 else "Fake" for c in pivot.columns]

    print(f"\n  Date distribution (year-month, all splits combined):")
    for period, row in pivot.iterrows():
        real = row.get("Real", 0)
        fake = row.get("Fake", 0)
        print(f"    {period}:  Real: {real:>4}  Fake: {fake:>4}")


def main():
    print("Dataset Statistics")
    print("Label convention: 0 = Real, 1 = Fake")

    for name, adapter in DATASETS:
        _print_label_stats(name, adapter)

        if name in DATE_DATASETS:
            _print_date_distribution(name, DATE_DATASETS[name])

    print()


if __name__ == "__main__":
    main()
