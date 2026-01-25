import requests

urls = [
    "https://openrouter.ai/api/v1/chat/completions",
    "https://openrouter.ai/api/v1",
    "https://openrouter.ai/api/v1/models",
]

for u in urls:
    try:
        r = requests.get(u)
        print(f"GET {u}: {r.status_code}")
    except Exception as e:
        print(f"GET {u} error: {e}")
