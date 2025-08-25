import requests
import sys

def test_api(base_url):
    """Test the API endpoints"""
    print(f"Testing API at: {base_url}")
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health")
        health_response.raise_for_status()
        print("✅ Health check passed:", health_response.json())
    except Exception as e:
        print("❌ Health check failed:", str(e))
        return False
    
    print("\nAPI is working correctly! 🚀")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <api_url>")
        print("Example: python test_api.py https://cyber-threat-detector-api.onrender.com")
        sys.exit(1)
    
    api_url = sys.argv[1]
    test_api(api_url)
