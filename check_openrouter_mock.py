import requests

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-or-v1-dummy",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://localhost",
    "X-Title": "Test"
}
data = {
    "model": "mistralai/mistral-7b-instruct:free",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True
}

try:
    response = requests.post(url, json=data, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
