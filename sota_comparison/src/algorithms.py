"""
SOTA baseline algorithms for fake news detection.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd


def get_algorithms():
    """
    Returns a dictionary of algorithm names and their pipeline configurations.
    """
    algorithms = {
        'SVM': Pipeline([
            ('bow', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', LinearSVC())
        ]),
        'LogisticRegression': Pipeline([
            ('bow', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', LogisticRegression())
        ]),
        'GradientBoosting': Pipeline([
            ('bow', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', GradientBoostingClassifier())
        ]),
        'DecisionTree': Pipeline([
            ('bow', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', DecisionTreeClassifier())
        ]),
        'KNN': Pipeline([
            ('bow', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])
    }
    return algorithms


def evaluate_model(y_true, y_pred):
    """
    Calculate all evaluation metrics.
    
    Returns:
        dict: Dictionary containing all metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def train_and_evaluate(algorithm_name, pipeline, train_data, val_data, test_data=None):
    """
    Train a model and evaluate it on validation (and optionally test) sets.
    
    Args:
        algorithm_name: Name of the algorithm
        pipeline: Sklearn pipeline
        train_data: (X_train, y_train) tuple
        val_data: (X_val, y_val) tuple
        test_data: Optional (X_test, y_test) tuple
    
    Returns:
        dict: Results containing metrics for val and test
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    print(f"  Training {algorithm_name}...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    print(f"  Evaluating on validation set...")
    val_pred = pipeline.predict(X_val)
    val_metrics = evaluate_model(y_val, val_pred)
    
    results = {
        'algorithm': algorithm_name,
        'val_metrics': val_metrics
    }
    
    # Evaluate on test set if provided
    if test_data is not None:
        X_test, y_test = test_data
        print(f"  Evaluating on test set...")
        test_pred = pipeline.predict(X_test)
        test_metrics = evaluate_model(y_test, test_pred)
        results['test_metrics'] = test_metrics
    
    return results


def print_results(results, dataset_name):
    """Pretty print results for a dataset"""
    print(f"\n{'='*70}")
    print(f"Results for {dataset_name} - {results['algorithm']}")
    print(f"{'='*70}")
    
    # Validation results
    print("\n📊 VALIDATION SET:")
    vm = results['val_metrics']
    print(f"  Accuracy : {vm['accuracy']:.4f}")
    print(f"  Precision: {vm['precision']:.4f}")
    print(f"  Recall   : {vm['recall']:.4f}")
    print(f"  F1-Score : {vm['f1']:.4f}")
    
    # Test results if available
    if 'test_metrics' in results:
        print("\n🧪 TEST SET:")
        tm = results['test_metrics']
        print(f"  Accuracy : {tm['accuracy']:.4f}")
        print(f"  Precision: {tm['precision']:.4f}")
        print(f"  Recall   : {tm['recall']:.4f}")
        print(f"  F1-Score : {tm['f1']:.4f}")
