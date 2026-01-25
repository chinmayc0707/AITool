import requests

url = "https://openrouter.ai/api/v1/chat/completions"
try:
    response = requests.post(url, json={})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

url_get = "https://openrouter.ai/api/v1/chat/completions"
try:
    response = requests.get(url_get)
    print(f"GET Status Code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
