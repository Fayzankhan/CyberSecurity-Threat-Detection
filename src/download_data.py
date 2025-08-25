import os
import requests
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL to the specified filename"""
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def main():
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # URLs for NSL-KDD dataset
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/NSL_KDD"
    files = {
        "KDDTrain+.txt": f"{base_url}/KDDTrain%2B.txt",
        "KDDTest+.txt": f"{base_url}/KDDTest%2B.txt"
    }
    
    # Download files
    for filename, url in files.items():
        output_file = data_dir / filename
        if not output_file.exists():
            download_file(url, output_file)
        else:
            print(f"{filename} already exists")

if __name__ == "__main__":
    main()
