import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not found. Will use RandomForest only.")

def train_models():
    print("Loading processed data...")
    X = np.load('datasets/X_processed.npy')
    y = np.load('datasets/y_processed.npy')
    
    # Due to the large dataset size (4M rows), we'll sample 500k for training 
    # to avoid excessive memory usage and training time while still being robust.
    # 500k is more than enough for these types of tree models.
    sample_size = min(500000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    
    results = {}
    
    # Random Forest - Shallow tree for low latency
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_preds),
        'precision': precision_score(y_test, rf_preds),
        'recall': recall_score(y_test, rf_preds),
        'f1': f1_score(y_test, rf_preds)
    }
    joblib.dump(rf, 'models/random_forest_model.joblib')
    
    if XGB_AVAILABLE:
        print("Training XGBoost...")
        # Shallow XGBoost for efficiency
        xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_test, xgb_preds),
            'precision': precision_score(y_test, xgb_preds),
            'recall': recall_score(y_test, xgb_preds),
            'f1': f1_score(y_test, xgb_preds)
        }
        joblib.dump(xgb, 'models/xgboost_model.joblib')
    
    print("\nTraining Results:")
    for model, metrics in results.items():
        print(f"--- {model} ---")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
    
    return results

if __name__ == "__main__":
    if os.path.exists('datasets/X_processed.npy'):
        train_models()
    else:
        print("Processed data not found. Run preprocess.py first.")
