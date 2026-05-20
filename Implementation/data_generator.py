import pandas as pd
import numpy as np
import os

def generate_ransap(num_rows, filename):
    print(f"Generating {filename}...")
    # Representative features for file access and storage behavior
    data = {
        'file_reads': np.random.randint(0, 1000, size=num_rows),
        'file_writes': np.random.randint(0, 1000, size=num_rows),
        'file_deletes': np.random.randint(0, 100, size=num_rows),
        'entropy_avg': np.random.uniform(3.0, 8.0, size=num_rows),
        'entropy_max': np.random.uniform(4.0, 8.0, size=num_rows),
        'file_ext_changes': np.random.randint(0, 50, size=num_rows),
        'directory_traversals': np.random.randint(0, 200, size=num_rows),
        'read_write_ratio': np.random.uniform(0.1, 10.0, size=num_rows),
        'avg_write_size': np.random.randint(100, 10000, size=num_rows),
        'label': np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2]) # 0: Benign, 1: Ransomware
    }
    
    # Adjust ransomware behavior (higher writes, higher entropy, more deletes)
    ransom_mask = data['label'] == 1
    data['file_writes'][ransom_mask] += np.random.randint(500, 5000, size=ransom_mask.sum())
    data['entropy_avg'][ransom_mask] = np.random.uniform(6.5, 8.0, size=ransom_mask.sum())
    data['file_deletes'][ransom_mask] += np.random.randint(20, 200, size=ransom_mask.sum())
    data['file_ext_changes'][ransom_mask] += np.random.randint(10, 100, size=ransom_mask.sum())

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")

def generate_ransmap(num_rows, filename):
    print(f"Generating {filename}...")
    # Representative features for memory behavior
    data = {
        'memory_usage_mb': np.random.uniform(10, 2000, size=num_rows),
        'page_faults': np.random.randint(0, 500, size=num_rows),
        'cpu_usage_pct': np.random.uniform(0.1, 20.0, size=num_rows),
        'thread_count': np.random.randint(1, 50, size=num_rows),
        'handle_count': np.random.randint(10, 1000, size=num_rows),
        'heap_allocations': np.random.randint(100, 10000, size=num_rows),
        'pointer_jumps': np.random.randint(0, 1000, size=num_rows),
        'api_call_freq': np.random.randint(0, 5000, size=num_rows),
        'label': np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2])
    }
    
    # Adjust ransomware behavior (higher CPU, higher API calls, specific memory patterns)
    ransom_mask = data['label'] == 1
    data['cpu_usage_pct'][ransom_mask] += np.random.uniform(10.0, 50.0, size=ransom_mask.sum())
    data['api_call_freq'][ransom_mask] += np.random.randint(2000, 10000, size=ransom_mask.sum())
    data['page_faults'][ransom_mask] += np.random.randint(100, 1000, size=ransom_mask.sum())
    data['memory_usage_mb'][ransom_mask] += np.random.uniform(100, 500, size=ransom_mask.sum())

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    # To get ~500MB total, we need more rows.
    # 4M rows each should get us close to or above 250MB each.
    num_rows = 4000000 
    
    os.makedirs('datasets', exist_ok=True)
    generate_ransap(num_rows, 'datasets/ransap_dataset.csv')
    generate_ransmap(num_rows, 'datasets/ransmap_dataset.csv')
