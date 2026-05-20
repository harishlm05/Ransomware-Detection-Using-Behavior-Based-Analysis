import numpy as np
import pandas as pd
import joblib
import os
import warnings
import random
from tabulate import tabulate

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

def load_inference_assets():
    scaler = joblib.load('models/scaler.joblib')
    if os.path.exists('models/xgboost_model.joblib'):
        model =  joblib.load('models/xgboost_model.joblib')
        model_name = "XGBoost"
    else:
        model = joblib.load('models/random_forest_model.joblib')
        model_name = "Random Forest"
    return scaler, model, model_name

def load_samples_from_dataset(n_samples=5):
    """Loads random samples from the fused datasets."""
    sap_path = 'datasets/ransap_dataset.csv'
    smap_path = 'datasets/ransmap_dataset.csv'
    
    if not (os.path.exists(sap_path) and os.path.exists(smap_path)):
        print("Dataset files not found. Using default scenarios.")
        return None, None

    df_sap = pd.read_csv(sap_path)
    df_smap = pd.read_csv(smap_path)
    
    # Sync and fuse (same logic as preprocess.py)
    min_len = min(len(df_sap), len(df_smap))
    df_sap = df_sap.iloc[:min_len]
    df_smap = df_smap.iloc[:min_len]
    
    # Check labels to pick a variety
    df_combined = pd.concat([
        df_sap.drop(columns=['label']).add_prefix('sap_'),
        df_smap.drop(columns=['label']).add_prefix('smap_'),
        pd.Series((df_sap['label'] | df_smap['label']).astype(int), name='ActualLabel')
    ], axis=1)
    df_combined['User_ID'] = [f"User_{i+1001}" for i in range(len(df_combined))]
    
    # Pick some benign and some ransomware
    benign = df_combined[df_combined['ActualLabel'] == 0].sample(min(n_samples//2 + 1, len(df_combined)))
    malicious = df_combined[df_combined['ActualLabel'] == 1].sample(min(n_samples//2, len(df_combined)))
    
    samples = pd.concat([benign, malicious]).sample(frac=1).reset_index(drop=True)
    return samples.drop(columns=['ActualLabel']), samples['ActualLabel']

def display_results(behavior_data, intelligence_data):
    print("\n" + "="*50)
    print(" TABLE 1: USER BEHAVIORAL METRICS (System Raw Data)")
    print("="*50)
    print(tabulate(behavior_data, headers='keys', tablefmt='psql', showindex=False))
    
    print("\n" + "="*50)
    print(" TABLE 2: AI CONTAINMENT INTELLIGENCE (Security Decisions)")
    print("="*50)
    print(tabulate(intelligence_data, headers='keys', tablefmt='psql', showindex=False))

def analyze_behavior(input_data, scaler, model, threshold=0.7):
    input_arr = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_arr)
    prob = model.predict_proba(scaled_input)[0][1]
    
    risk_level = "LOW"
    action = "SAFE / NO PROBLEM"
    if prob > threshold:
        risk_level = "CRITICAL"
        action = "BLOCK USER / ISOLATE"
    elif prob > 0.3:
        risk_level = "MEDIUM"
        action = "STRICT MONITORING"
        
    return prob, risk_level, action

if __name__ == "__main__":
    print("Initializing Ransomware Containment System...")
    scaler, model, m_name = load_inference_assets()
    print(f"Loaded {m_name} model for real-time intelligence.")
    
    dataset_samples, actual_labels = load_samples_from_dataset(n_samples=6)
    
    behavior_history = []
    intelligence_history = []
    
    if dataset_samples is not None:
        print(f"Sampling {len(dataset_samples)} activities from datasets...")
        for i in range(len(dataset_samples)):
            sample = dataset_samples.iloc[i]
            actual = actual_labels.iloc[i]
            
            # Predict
            features_for_model = sample.drop('User_ID').values
            prob, risk, action = analyze_behavior(features_for_model, scaler, model)
            
            user_id = sample['User_ID']
            
            # Table 1: All 17 Features
            behavior_history.append({
                'User': user_id,
                'Reads': f"{sample.get('sap_file_reads', 0):.0f}",
                'Writes': f"{sample.get('sap_file_writes', 0):.0f}",
                'Entr': f"{sample.get('sap_entropy_avg', 0):.2f}",
                'Ext': f"{sample.get('sap_file_ext_changes', 0):.0f}",
                'Mem_MB': f"{sample.get('smap_memory_usage_mb', 0):.1f}",
                'CPU%': f"{sample.get('smap_cpu_usage_pct', 0):.1f}",
                'API': f"{sample.get('smap_api_call_freq', 0):.0f}",
                'Thr': f"{sample.get('smap_thread_count', 0):.0f}",
                'Hnd': f"{sample.get('smap_handle_count', 0):.0f}"
            })
            
            # Table 2: AI Response
            intelligence_history.append({
                'User': user_id,
                'Actual Type': 'MALICIOUS' if actual == 1 else 'BENIGN',
                'Risk Score': f"{prob:.4f}",
                'Security Level': risk,
                'DECISION / INSTRUCTION': action
            })
    else:
        print("Note: Dataset not found, simulation mode active.")

    if behavior_history:
        display_results(behavior_history, intelligence_history)
    
    # Final Decision for the last input
    if intelligence_history:
        last_res = intelligence_history[-1]
        print(f"\n>>> FINAL SECURITY REPORT FOR {last_res['User']} <<<")
        print(f"Result: {last_res['DECISION / INSTRUCTION']}")
        print(f"Confidence: {float(last_res['Risk Score'])*100:.2f}%")
