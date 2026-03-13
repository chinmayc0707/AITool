import sys
from unittest.mock import MagicMock, patch
import time

class MockSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

# Mock streamlit before importing ai1
mock_st = MagicMock()
mock_st.session_state = MockSessionState()
mock_st.cache_resource = lambda func: func
def mock_cache_data(**kwargs):
    def decorator(func):
        # simple memoization for the mock
        cache = {}
        def wrapper(*args, **kw):
            key = str(args) + str(kw)
            if key not in cache:
                cache[key] = func(*args, **kw)
            return cache[key]
        return wrapper
    return decorator
mock_st.cache_data = mock_cache_data
sys.modules["streamlit"] = mock_st
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["pdfplumber"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["langchain"] = MagicMock()
sys.modules["langchain.text_splitter"] = MagicMock()
sys.modules["langchain.embeddings"] = MagicMock()
sys.modules["langchain.vectorstores"] = MagicMock()
sys.modules["langchain.docstore"] = MagicMock()
sys.modules["faiss"] = MagicMock()

import ai1

def run_benchmark():
    # Mock requests.get to simulate a slow network call
    import requests
    with patch('requests.get') as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
        mock_get.return_value = mock_resp

        # Simulate network latency
        def side_effect(*args, **kwargs):
            time.sleep(0.05)
            return mock_resp
        mock_get.side_effect = side_effect

        start_time = time.time()
        for _ in range(20):
            ai1.get_ollama_models("http://fakehost")
        end_time = time.time()

        print(f"Time taken for 20 calls: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
