import sys
import unittest
from unittest.mock import MagicMock, patch
import io
import os

class MockSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

# Mock all dependencies before importing ai1
mock_st = MagicMock()
mock_st.session_state = MockSessionState()
mock_st.cache_resource = lambda func: func
mock_st.cache_data = lambda **kwargs: lambda func: func
mock_st.spinner = MagicMock(return_value=MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None))
mock_st.sidebar = MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None)
sys.modules["streamlit"] = mock_st

sys.modules["requests"] = MagicMock()
sys.modules["pdfplumber"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["faiss"] = MagicMock()

# Mock langchain and its submodules
mock_langchain = MagicMock()
sys.modules["langchain"] = mock_langchain
sys.modules["langchain.text_splitter"] = MagicMock()
sys.modules["langchain.embeddings"] = MagicMock()
sys.modules["langchain.vectorstores"] = MagicMock()
sys.modules["langchain.docstore"] = MagicMock()

# Now import the module under test
import ai1

class TestFix(unittest.TestCase):
    def setUp(self):
        # Reset session state
        mock_st.session_state = MockSessionState()
        ai1.init_session()

        # Mock FAISS and Embeddings
        self.mock_faiss_instance = MagicMock()
        ai1.FAISS = MagicMock(return_value=self.mock_faiss_instance)
        ai1.HuggingFaceEmbeddings = MagicMock()

    def test_extract_pdf_optimization(self):
        """Verify _extract_pdf returns correct text with new implementation."""
        page = MagicMock()
        page.extract_text.return_value = "Line 1"
        page.extract_tables.return_value = [[["A", "B"], ["C", "D"]]]

        pdf_mock = MagicMock()
        pdf_mock.pages = [page]

        with patch("ai1.pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = pdf_mock

            result = ai1._extract_pdf("dummy.pdf")

            self.assertEqual(result, "Line 1\nA | B\nC | D")

    def test_init_session(self):
        """Test that init_session initializes expected keys."""
        ai1.init_session()
        self.assertIn("messages", mock_st.session_state)
        self.assertIn("processed_hashes", mock_st.session_state)
        self.assertIn("hash2file", mock_st.session_state)

    def test_file_type(self):
        self.assertEqual(ai1._file_type("test.pdf"), "PDF")
        self.assertEqual(ai1._file_type("test.png"), "IMAGE")

if __name__ == "__main__":
    unittest.main()
