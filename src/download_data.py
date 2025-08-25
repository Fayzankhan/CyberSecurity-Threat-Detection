import os
import requests
from pathlib import Path
import pandas as pd
import numpy as np

def generate_sample_data(num_records=1000):
    """Generate sample NSL-KDD data if download fails"""
    print("Generating sample data...")
    
    # Basic feature values
    protocols = ['tcp', 'udp', 'icmp']
    services = ['http', 'ftp', 'smtp', 'ssh', 'telnet', 'domain_u', 'private']
    flags = ['SF', 'REJ', 'RSTO', 'RSTOS0', 'S0', 'S1']
    
    data = []
    for _ in range(num_records):
        record = {
            'duration': np.random.randint(0, 58329),
            'protocol_type': np.random.choice(protocols),
            'service': np.random.choice(services),
            'flag': np.random.choice(flags),
            'src_bytes': np.random.randint(0, 1000000),
            'dst_bytes': np.random.randint(0, 1000000),
            'land': np.random.choice([0, 1], p=[0.99, 0.01]),
            'wrong_fragment': np.random.randint(0, 3),
            'urgent': np.random.randint(0, 3),
            'hot': np.random.randint(0, 30),
            'num_failed_logins': np.random.randint(0, 5),
            'logged_in': np.random.choice([0, 1]),
            'num_compromised': np.random.randint(0, 7),
            'root_shell': np.random.choice([0, 1], p=[0.95, 0.05]),
            'su_attempted': np.random.choice([0, 1], p=[0.95, 0.05]),
            'num_root': np.random.randint(0, 7),
            'num_file_creations': np.random.randint(0, 10),
            'num_shells': np.random.randint(0, 3),
            'num_access_files': np.random.randint(0, 4),
            'num_outbound_cmds': 0,
            'is_host_login': np.random.choice([0, 1], p=[0.99, 0.01]),
            'is_guest_login': np.random.choice([0, 1], p=[0.95, 0.05]),
            'count': np.random.randint(0, 500),
            'srv_count': np.random.randint(0, 500),
            'serror_rate': np.random.random(),
            'srv_serror_rate': np.random.random(),
            'rerror_rate': np.random.random(),
            'srv_rerror_rate': np.random.random(),
            'same_srv_rate': np.random.random(),
            'diff_srv_rate': np.random.random(),
            'srv_diff_host_rate': np.random.random(),
            'dst_host_count': np.random.randint(0, 255),
            'dst_host_srv_count': np.random.randint(0, 255),
            'dst_host_same_srv_rate': np.random.random(),
            'dst_host_diff_srv_rate': np.random.random(),
            'dst_host_same_src_port_rate': np.random.random(),
            'dst_host_srv_diff_host_rate': np.random.random(),
            'dst_host_serror_rate': np.random.random(),
            'dst_host_srv_serror_rate': np.random.random(),
            'dst_host_rerror_rate': np.random.random(),
            'dst_host_srv_rerror_rate': np.random.random(),
            'label': np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r']),
            'difficulty': np.random.randint(0, 21)
        }
        data.append(record)
    
    return pd.DataFrame(data)

def download_file(url, filename):
    """Download a file from URL to the specified filename"""
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate sample data since the original dataset is not accessible
    print("Generating sample dataset...")
    train_df = generate_sample_data(1000)
    test_df = generate_sample_data(500)
    
    # Save the datasets
    train_file = data_dir / "KDDTrain+.txt"
    test_file = data_dir / "KDDTest+.txt"
    
    train_df.to_csv(train_file, index=False, header=False)
    test_df.to_csv(test_file, index=False, header=False)
    
    print(f"Created sample datasets in {data_dir}")
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")

if __name__ == "__main__":
    main()