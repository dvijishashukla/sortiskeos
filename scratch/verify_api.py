import requests
try:
    r = requests.get('http://localhost:8000/dashboard/stats')
    r.raise_for_status()
    data = r.json()
    print(f"Keys: {list(data.keys())}")
    print(f"Category: {data.get('suggestion', {}).get('category')}")
except Exception as e:
    print(f"Error: {e}")
