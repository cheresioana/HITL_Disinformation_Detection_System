"""
Dataset adapters for loading and normalizing different datasets.
Each adapter returns DataFrames with standardized columns: 'text', 'label' (0=real, 1=fake)
"""
import pandas as pd
from pathlib import Path


class CovidDatasetAdapter:
    """Adapter for the COVID-19 fake news dataset"""
    
    def __init__(self, base_path=None):
        if base_path is None:
            # Default to datasets/covid relative to this file's parent directory
            base_path = Path('datasets') / 'covid'
        self.base_path = Path(base_path)
    
    def load_train(self):
        """Load training data"""
        df = pd.read_csv(self.base_path / 'train.csv')
        return self._normalize(df)
    
    def load_val(self):
        """Load validation data"""
        df = pd.read_csv(self.base_path / 'eval.csv')
        return self._normalize(df)
    
    def load_test(self):
        """Load test data if available"""
        test_path = self.base_path / 'test.csv'
        if test_path.exists():
            df = pd.read_csv(test_path)
            return self._normalize(df)
        return None
    
    def _normalize(self, df):
        """Normalize to standard format"""
        # Rename tweet to text
        df = df.rename(columns={'tweet': 'text'})
        
        # Convert label strings to 0/1
        label_map = {'real': 0, 'fake': 1}
        df['label'] = df['label'].map(label_map)
        
        # Keep only needed columns
        df = df[['text', 'label']].copy()
        
        return df


class MindbugsDatasetAdapter:
    """Adapter for the Mindbugs dataset"""
    
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path('datasets') / 'mindbugs_updated'
        self.base_path = Path(base_path)
    
    def load_train(self):
        """Load training data"""
        df = pd.read_csv(self.base_path / 'train.csv')
        return self._normalize(df)
    
    def load_val(self):
        """Load validation data"""
        df = pd.read_csv(self.base_path / 'validation_df.csv')
        return self._normalize(df)
    
    def load_test(self):
        """Load test data"""
        df = pd.read_csv(self.base_path / 'evaluation.csv')
        return self._normalize(df)
    
    def _normalize(self, df):
        """Normalize to standard format"""
        # Keep text column (already named 'text')
        # Convert label boolean/string to 0/1
        # True/False or "True"/"False" -> 0 (real), 1 (fake)
        if df['label'].dtype == bool:
            df['label'] = (~df['label']).astype(int)  # True=real=0, False=fake=1
        else:
            # String values "True"/"False"
            label_map = {'True': 0, 'False': 1, True: 0, False: 1}
            df['label'] = df['label'].map(label_map)
        
        # Keep only needed columns
        df = df[['text', 'label']].copy()
        
        return df


def load_covid_dataset(include_test=False):
    """
    Convenience function to load COVID dataset.
    
    Args:
        include_test: If True, also load test set
    
    Returns:
        tuple: (train_df, val_df, test_df) if include_test else (train_df, val_df)
    """
    adapter = CovidDatasetAdapter()
    train = adapter.load_train()
    val = adapter.load_val()
    
    print(f"COVID Dataset Loaded:")
    print(f"  Train size: {len(train)} (Real: {(train['label'] == 0).sum()}, Fake: {(train['label'] == 1).sum()})")
    print(f"  Val size: {len(val)} (Real: {(val['label'] == 0).sum()}, Fake: {(val['label'] == 1).sum()})")
    
    if include_test:
        test = adapter.load_test()
        print(f"  Test size: {len(test)} (Real: {(test['label'] == 0).sum()}, Fake: {(test['label'] == 1).sum()})")
        return train, val, test
    
    return train, val


class LiarDatasetAdapter:
    """Adapter for the LIAR dataset"""
    
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path('datasets') / 'liar'
        self.base_path = Path(base_path)
    
    def load_train(self):
        """Load training data"""
        df = pd.read_csv(self.base_path / 'train.tsv', sep='\t', header=None)
        return self._normalize(df)
    
    def load_val(self):
        """Load validation data"""
        df = pd.read_csv(self.base_path / 'valid.tsv', sep='\t', header=None)
        return self._normalize(df)
    
    def load_test(self):
        """Load test data"""
        df = pd.read_csv(self.base_path / 'test.tsv', sep='\t', header=None)
        return self._normalize(df)
    
    def _normalize(self, df):
        """
        Normalize to standard format.
        LIAR has 6 labels, we map to binary:
        - pants-fire, false, barely-true → Fake (1)
        - half-true, mostly-true, true → Real (0)
        """
        # Column 1 is the label, Column 2 is the statement
        df.columns = ['id', 'label_orig', 'text'] + [f'col_{i}' for i in range(3, len(df.columns))]
        
        # Map to binary labels
        fake_labels = ['pants-fire', 'false', 'barely-true']
        real_labels = ['half-true', 'mostly-true', 'true']
        
        df['label'] = df['label_orig'].apply(
            lambda x: 1 if x in fake_labels else (0 if x in real_labels else -1)
        )
        
        # Remove any rows with invalid labels (shouldn't happen)
        df = df[df['label'] != -1]
        
        # Keep only text and label
        df = df[['text', 'label']].copy()
        
        return df


def load_mindbugs_dataset(include_test=False):
    """
    Convenience function to load Mindbugs dataset.

    Args:
        include_test: If True, also load test set

    Returns:
        tuple: (train_df, val_df, test_df) if include_test else (train_df, val_df)
    """
    adapter = MindbugsDatasetAdapter()
    train = adapter.load_train()
    val = adapter.load_val()

    print(f"Mindbugs Dataset Loaded:")
    print(f"  Train size: {len(train)} (Real: {(train['label'] == 0).sum()}, Fake: {(train['label'] == 1).sum()})")
    print(f"  Val size: {len(val)} (Real: {(val['label'] == 0).sum()}, Fake: {(val['label'] == 1).sum()})")

    if include_test:
        test = adapter.load_test()
        print(f"  Test size: {len(test)} (Real: {(test['label'] == 0).sum()}, Fake: {(test['label'] == 1).sum()})")
        return train, val, test

    return train, val


class MindbugsRoDatasetAdapter(MindbugsDatasetAdapter):
    """Adapter for the Romanian-translated Mindbugs dataset.
    Identical to MindbugsDatasetAdapter but reads from datasets/mindbugs_ro/."""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path('datasets') / 'mindbugs_ro'
        super().__init__(base_path=base_path)


def load_mindbugs_ro_dataset(include_test=False):
    """
    Convenience function to load Romanian Mindbugs dataset.

    Args:
        include_test: If True, also load test set

    Returns:
        tuple: (train_df, val_df, test_df) if include_test else (train_df, val_df)
    """
    adapter = MindbugsRoDatasetAdapter()
    train = adapter.load_train()
    val = adapter.load_val()

    print(f"Mindbugs-RO Dataset Loaded:")
    print(f"  Train size: {len(train)} (Real: {(train['label'] == 0).sum()}, Fake: {(train['label'] == 1).sum()})")
    print(f"  Val size: {len(val)} (Real: {(val['label'] == 0).sum()}, Fake: {(val['label'] == 1).sum()})")

    if include_test:
        test = adapter.load_test()
        print(f"  Test size: {len(test)} (Real: {(test['label'] == 0).sum()}, Fake: {(test['label'] == 1).sum()})")
        return train, val, test

    return train, val


class WELFakeDatasetAdapter:
    """Adapter for the WELFake dataset"""
    
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path('datasets') / 'welfake'
        self.base_path = Path(base_path)
    
    def load_train(self):
        """Load training data"""
        df = pd.read_csv(self.base_path / 'train_welfake.csv')
        return self._normalize(df)
    
    def load_val(self):
        """Load validation data"""
        df = pd.read_csv(self.base_path / 'val_welfake.csv')
        return self._normalize(df)
    
    def load_test(self):
        """Load test data"""
        df = pd.read_csv(self.base_path / 'test_welfake.csv')
        return self._normalize(df)
    
    def _normalize(self, df):
        """Normalize to standard format"""
        # WELFake has: title, text, label (0=fake, 1=real)
        # We'll use title column and rename it to text
        # Need to invert label to match our convention (0=real, 1=fake)
        df = df[['title', 'label']].copy()
        
        # Rename title to text
        df = df.rename(columns={'title': 'text'})
        
        # Invert label: real (1) -> 0, fake (0) -> 1
        df['label'] = 1 - df['label']
        
        # Drop rows with missing text
        df = df.dropna(subset=['text'])
        
        return df


def load_liar_dataset(include_test=False):
    """
    Convenience function to load LIAR dataset.
    
    Args:
        include_test: If True, also load test set
    
    Returns:
        tuple: (train_df, val_df, test_df) if include_test else (train_df, val_df)
    """
    adapter = LiarDatasetAdapter()
    train = adapter.load_train()
    val = adapter.load_val()
    
    print(f"LIAR Dataset Loaded:")
    print(f"  Train size: {len(train)} (Real: {(train['label'] == 0).sum()}, Fake: {(train['label'] == 1).sum()})")
    print(f"  Val size: {len(val)} (Real: {(val['label'] == 0).sum()}, Fake: {(val['label'] == 1).sum()})")
    
    if include_test:
        test = adapter.load_test()
        print(f"  Test size: {len(test)} (Real: {(test['label'] == 0).sum()}, Fake: {(test['label'] == 1).sum()})")
        return train, val, test
    
    return train, val


class FakeNewsNetDatasetAdapter:
    """Adapter for the FakeNewsNet dataset"""
    
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path('datasets') / 'fake_news_net'
        self.base_path = Path(base_path)
    
    def load_train(self):
        """Load training data"""
        df = pd.read_csv(self.base_path / 'train_fakenewsnet.csv')
        return self._normalize(df)
    
    def load_val(self):
        """Load validation data"""
        df = pd.read_csv(self.base_path / 'val_fakenewsnet.csv')
        return self._normalize(df)
    
    def load_test(self):
        """Load test data"""
        df = pd.read_csv(self.base_path / 'test_fakenewsnet.csv')
        return self._normalize(df)
    
    def _normalize(self, df):
        """Normalize to standard format"""
        # FakeNewsNet has: title, news_url, source_domain, tweet_num, real
        # We'll use title column and rename it to text
        # real: 1=real, 0=fake -> need to invert to match our convention (0=real, 1=fake)
        df = df[['title', 'real']].copy()
        
        # Rename title to text
        df = df.rename(columns={'title': 'text', 'real': 'label'})
        
        # Invert label: real (1) -> 0, fake (0) -> 1
        df['label'] = 1 - df['label']
        
        # Drop rows with missing text
        df = df.dropna(subset=['text'])
        
        return df


def load_welfake_dataset(include_test=False):
    """
    Convenience function to load WELFake dataset.
    
    Args:
        include_test: If True, also load test set
    
    Returns:
        tuple: (train_df, val_df, test_df) if include_test else (train_df, val_df)
    """
    adapter = WELFakeDatasetAdapter()
    train = adapter.load_train()
    val = adapter.load_val()
    
    print(f"WELFake Dataset Loaded:")
    print(f"  Train size: {len(train)} (Real: {(train['label'] == 0).sum()}, Fake: {(train['label'] == 1).sum()})")
    print(f"  Val size: {len(val)} (Real: {(val['label'] == 0).sum()}, Fake: {(val['label'] == 1).sum()})")
    
    if include_test:
        test = adapter.load_test()
        print(f"  Test size: {len(test)} (Real: {(test['label'] == 0).sum()}, Fake: {(test['label'] == 1).sum()})")
        return train, val, test
    
    return train, val


def load_fakenewsnet_dataset(include_test=False):
    """
    Convenience function to load FakeNewsNet dataset.
    
    Args:
        include_test: If True, also load test set
    
    Returns:
        tuple: (train_df, val_df, test_df) if include_test else (train_df, val_df)
    """
    adapter = FakeNewsNetDatasetAdapter()
    train = adapter.load_train()
    val = adapter.load_val()
    
    print(f"FakeNewsNet Dataset Loaded:")
    print(f"  Train size: {len(train)} (Real: {(train['label'] == 0).sum()}, Fake: {(train['label'] == 1).sum()})")
    print(f"  Val size: {len(val)} (Real: {(val['label'] == 0).sum()}, Fake: {(val['label'] == 1).sum()})")
    
    if include_test:
        test = adapter.load_test()
        print(f"  Test size: {len(test)} (Real: {(test['label'] == 0).sum()}, Fake: {(test['label'] == 1).sum()})")
        return train, val, test
    
    return train, val


if __name__ == '__main__':
    # Test the adapter
    train, val = load_welfake_dataset()
    print("\nSample from train:")
    print(train.head(3))
    print("\nSample from val:")
    print(val.head(3))
