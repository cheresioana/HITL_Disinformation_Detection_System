"""
Run all baseline algorithms on all datasets and save results.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sota_comparison.src.adapters import load_covid_dataset, load_mindbugs_dataset, load_liar_dataset, load_fakenewsnet_dataset
from sota_comparison.src.algorithms import get_algorithms, train_and_evaluate, print_results

# Download stopwords if needed
try:
    stops = set(stopwords.words("english"))
except:
    nltk.download('stopwords', quiet=True)
    stops = set(stopwords.words("english"))


def cleantext(string):
    """Clean text data"""
    # Handle NaN/None values
    if pd.isna(string) or string is None:
        return ""
    
    # Convert to string if not already
    string = str(string)
    
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+", ' ', text)    
    text = re.sub(r"www(\S)+", ' ', text)
    text = re.sub(r"&", 'and', text)
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    text = text.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def prepare_dataset(train, val, test):
    """Clean and prepare dataset"""
    print("  Cleaning text data...")
    train['text'] = train['text'].map(lambda x: cleantext(x))
    val['text'] = val['text'].map(lambda x: cleantext(x))
    test['text'] = test['text'].map(lambda x: cleantext(x))
    
    return (
        (train['text'], train['label']),
        (val['text'], val['label']),
        (test['text'], test['label'])
    )


def run_experiments_on_dataset(dataset_name, train, val, test):
    """
    Run all algorithms on a single dataset.
    
    Returns:
        list: Results for all algorithms
    """
    print(f"\n{'#'*70}")
    print(f"# Running experiments on {dataset_name}")
    print(f"{'#'*70}")
    
    # Prepare data
    train_data, val_data, test_data = prepare_dataset(train, val, test)
    
    # Get all algorithms
    algorithms = get_algorithms()
    
    # Run each algorithm
    all_results = []
    for alg_name, pipeline in algorithms.items():
        print(f"\n{'-'*70}")
        print(f"Algorithm: {alg_name}")
        print(f"{'-'*70}")
        
        results = train_and_evaluate(
            alg_name, 
            pipeline, 
            train_data, 
            val_data, 
            test_data
        )
        results['dataset'] = dataset_name
        all_results.append(results)
        
        # Print results
        print_results(results, dataset_name)
    
    return all_results


def save_results(all_results, output_dir='results/sota_results'):
    """Save results to JSON and CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed JSON
    json_path = output_path / 'all_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Detailed results saved to: {json_path}")
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        # Validation metrics
        summary_data.append({
            'dataset': result['dataset'],
            'algorithm': result['algorithm'],
            'split': 'validation',
            'accuracy': result['val_metrics']['accuracy'],
            'precision': result['val_metrics']['precision'],
            'recall': result['val_metrics']['recall'],
            'f1': result['val_metrics']['f1']
        })
        
        # Test metrics
        if 'test_metrics' in result:
            summary_data.append({
                'dataset': result['dataset'],
                'algorithm': result['algorithm'],
                'split': 'test',
                'accuracy': result['test_metrics']['accuracy'],
                'precision': result['test_metrics']['precision'],
                'recall': result['test_metrics']['recall'],
                'f1': result['test_metrics']['f1']
            })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = output_path / 'summary_results.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"✅ Summary CSV saved to: {csv_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))


def main():
    """Main execution function"""
    print("="*70)
    print("SOTA BASELINE COMPARISON")
    print("="*70)
    
    all_results = []
    
    # ========== COVID DATASET ==========
    print("\n📦 Loading COVID dataset...")
    covid_train, covid_val, covid_test = load_covid_dataset(include_test=True)
    covid_results = run_experiments_on_dataset(
        'COVID-19',
        covid_train,
        covid_val,
        covid_test
    )
    all_results.extend(covid_results)
    
    # ========== MINDBUGS DATASET ==========
    print("\n📦 Loading Mindbugs dataset...")
    mb_train, mb_val, mb_test = load_mindbugs_dataset(include_test=True)
    
    # Run baseline algorithms
    mb_results = run_experiments_on_dataset(
        'Mindbugs',
        mb_train,
        mb_val,
        mb_test
    )
    all_results.extend(mb_results)
    
    # ========== LIAR DATASET ==========
    print("\n📦 Loading LIAR dataset...")
    liar_train, liar_val, liar_test = load_liar_dataset(include_test=True)
    liar_results = run_experiments_on_dataset(
        'LIAR',
        liar_train,
        liar_val,
        liar_test
    )
    all_results.extend(liar_results)

    # ========== FAKENEWSNET DATASET ==========
    print("\n📦 Loading FakeNewsNet dataset...")
    fnn_train, fnn_val, fnn_test = load_fakenewsnet_dataset(include_test=True)
    fnn_results = run_experiments_on_dataset(
        'FakeNewsNet',
        fnn_train,
        fnn_val,
        fnn_test
    )
    all_results.extend(fnn_results)
    
    # ========== SAVE RESULTS ==========
    save_results(all_results)
    
    print(f"\n{'='*70}")
    print("✅ All experiments completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
