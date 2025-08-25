import requests
import sys
import pandas as pd
from pathlib import Path

def load_sample_data():
    """Load sample data for testing"""
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample.csv"
    if not data_path.exists():
        print("❌ Sample data not found at:", data_path)
        return None
    return pd.read_csv(data_path)

def test_predictions(base_url):
    """Test both binary and multiclass prediction endpoints"""
    print(f"Testing predictions at: {base_url}")
    
    # Load sample data
    df = load_sample_data()
    if df is None:
        return False
    
    # Prepare sample records
    records = df.head(2).to_dict(orient="records")
    payload = {"records": records}
    
    # Test binary classification
    print("\nTesting binary classification...")
    try:
        response = requests.post(f"{base_url}/predict-batch", json=payload)
        response.raise_for_status()
        result = response.json()
        print("✅ Binary classification successful:")
        print("  - Predictions:", result["predictions"])
        print("  - Probabilities:", [f"{p:.3f}" for p in result["probabilities"]])
    except Exception as e:
        print("❌ Binary classification failed:", str(e))
        return False
    
    # Test multiclass classification
    print("\nTesting multiclass classification...")
    try:
        response = requests.post(f"{base_url}/predict-multiclass", json=payload)
        response.raise_for_status()
        result = response.json()
        print("✅ Multiclass classification successful:")
        print("  - Predictions:", result["predictions"])
        print("  - Confidence:", [f"{c:.3f}" if c is not None else "N/A" for c in result["confidence"]])
    except Exception as e:
        print("❌ Multiclass classification failed:", str(e))
        return False
    
    print("\nAll prediction tests passed! 🎉")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_predictions.py <api_url>")
        print("Example: python test_predictions.py https://cyber-threat-detector-api.onrender.com")
        sys.exit(1)
    
    api_url = sys.argv[1]
    test_predictions(api_url)
