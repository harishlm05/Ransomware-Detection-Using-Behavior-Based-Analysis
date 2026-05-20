import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess(sap_path, smap_path):
    print("Loading datasets...")
    df_sap = pd.read_csv(sap_path)
    df_smap = pd.read_csv(smap_path)
    
    # Feature Fusion: Combining RanSAP (storage) and RanSMAP (memory)
    # In a real scenario, we would join on Process ID or Timestamp.
    # Here we assume rows are synchronized observations and concatenate.
    # We'll use the minimum length to avoid NaN if lengths differ slightly.
    min_len = min(len(df_sap), len(df_smap))
    df_sap = df_sap.iloc[:min_len]
    df_smap = df_smap.iloc[:min_len]
    
    # Check for label consistency (simple fusion logic)
    # If either indicates ransomware, we mark it as such for training, or use a specific one.
    # We'll take the SAP label as primary if it exists, or bitwise OR.
    combined_label = (df_sap['label'] | df_smap['label']).astype(int)
    
    # Drop labels from individual dataframes before concatenation
    df_sap_features = df_sap.drop(columns=['label'])
    df_smap_features = df_smap.drop(columns=['label'])
    
    # Rename columns to avoid collision and clarify source
    df_sap_features.columns = [f'sap_{c}' for c in df_sap_features.columns]
    df_smap_features.columns = [f'smap_{c}' for c in df_smap_features.columns]
    
    fused_df = pd.concat([df_sap_features, df_smap_features], axis=1)
    fused_df['label'] = combined_label
    
    print(f"Fused dataset shape: {fused_df.shape}")
    
    # Cleaning: Fill missing (though synthetic shouldn't have any)
    fused_df = fused_df.fillna(0)
    
    return fused_df

def apply_normalization(df, save_scaler_path='models/scaler.joblib'):
    print("Applying normalization...")
    X = df.drop(columns=['label'])
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, save_scaler_path)
    print(f"Scaler saved to {save_scaler_path}")
    
    return X_scaled, y

def sliding_window_aggregation(X, window_size=5):
    """
    Simple sliding window to capture temporal patterns.
    Aggregates previous 'window_size' steps.
    For simplicity in this baseline, we'll use a rolling mean.
    """
    print(f"Applying sliding window (size={window_size})...")
    df_X = pd.DataFrame(X)
    X_rolled = df_X.rolling(window=window_size).mean().dropna()
    return X_rolled.values, window_size

if __name__ == "__main__":
    sap_path = 'datasets/ransap_dataset.csv'
    smap_path = 'datasets/ransmap_dataset.csv'
    
    if os.path.exists(sap_path) and os.path.exists(smap_path):
        fused_df = load_and_preprocess(sap_path, smap_path)
        X_scaled, y = apply_normalization(fused_df)
        
        # We'll save the processed data for training to avoid re-processing
        # But for 500MB, saving to CSV again might be slow. 
        # We can just return it or save as a compressed pickle/numpy.
        np.save('datasets/X_processed.npy', X_scaled)
        np.save('datasets/y_processed.npy', y.values)
        print("Processed data saved to datasets/X_processed.npy and datasets/y_processed.npy")
    else:
        print("Datasets not found. Run data_generator.py first.")
