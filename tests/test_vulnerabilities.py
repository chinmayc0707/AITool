import sys
import unittest
from unittest.mock import MagicMock, patch

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
mock_st.spinner = MagicMock(return_value=MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None))
sys.modules["streamlit"] = mock_st

# Mock dependencies
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

class TestVulnerabilities(unittest.TestCase):
    def setUp(self):
        # Reset session state
        mock_st.session_state.clear()
        mock_st.session_state.update({
            "processed_uploads": set(),
            "processed_hashes": set(),
            "source_files": set(),
            "vector_store": MagicMock()
        })

    def test_handle_chunked_read(self):
        """Test that _handle reads file in chunks to avoid DoS."""
        mock_file = MagicMock()
        mock_file.name = "large_file.pdf"
        mock_file.size = 10 * 1024 * 1024  # 10MB

        # Simulate read behavior: return data in chunks, then empty bytes
        mock_file.read.side_effect = [b"chunk" * 100, b""]

        # We need to mock _process so it doesn't fail
        with patch("ai1._process") as mock_process:
            mock_process.return_value = ([], "PDF")

            # Call _handle
            try:
                ai1._handle([mock_file])
            except Exception as e:
                # If reading fails because read() was called without args on the mock side_effect list
                # (which expects repeated calls), catch it.
                # But here we just want to inspect the calls.
                pass

            # Verify read was called with an argument (chunk size)
            read_calls = mock_file.read.mock_calls
            self.assertTrue(len(read_calls) > 0, "File should be read")

            # Check the first call
            first_call_args = read_calls[0].args
            if not first_call_args:
                self.fail(f"File.read() called without arguments (vulnerable to DoS). Calls: {read_calls}")

            chunk_size = first_call_args[0]
            self.assertLessEqual(chunk_size, 65536, f"Read chunk size {chunk_size} is too large")

if __name__ == "__main__":
    unittest.main()
