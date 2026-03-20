import sys
import unittest
from unittest.mock import MagicMock, patch
import io

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
# Mock cache_resource to return the function itself (identity decorator)
mock_st.cache_resource = lambda func: func
mock_st.spinner = MagicMock(return_value=MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None))
mock_st.sidebar = MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None)
sys.modules["streamlit"] = mock_st

# Also mock pdfplumber and pytesseract and PIL to avoid dependencies or file ops
sys.modules["requests"] = MagicMock()
sys.modules["pdfplumber"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["PIL"] = MagicMock()
# Mock langchain components
sys.modules["langchain"] = MagicMock()
sys.modules["langchain.text_splitter"] = MagicMock()
sys.modules["langchain.embeddings"] = MagicMock()
sys.modules["langchain.vectorstores"] = MagicMock()
sys.modules["langchain.docstore"] = MagicMock()
sys.modules["faiss"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

# Now import the module under test
import ai1

class TestFix(unittest.TestCase):
    def setUp(self):
        # Reset session state
        mock_st.session_state = MockSessionState()
        # Mock FAISS in ai1
        # ai1.FAISS is the class. We want to mock the instance it returns.
        self.mock_faiss_instance = MagicMock()
        # We need to patch the FAISS class inside ai1
        # Since we mocked the module 'langchain.vectorstores', ai1.FAISS is that mock.
        ai1.FAISS.return_value = self.mock_faiss_instance

        # Mock Embedding
        ai1.HuggingFaceEmbeddings.return_value = MagicMock()

    def test_init_session(self):
        """Test that init_session initializes expected state keys."""
        ai1.init_session()
        self.assertIn("messages", mock_st.session_state)
        self.assertIn("processed_hashes", mock_st.session_state)
        self.assertIn("provider", mock_st.session_state)

    def test_handle_upload(self):
        """Test that _handle processes upload properly."""
        # Setup session state
        ai1.init_session()

        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.size = 1234
        mock_file.getvalue.return_value = b"fake content"

        # Mock add_to_db to prevent file operations from actually happening
        with patch("ai1.add_to_db") as mock_add_to_db:
            # Also mock open so we don't write to disk
            with patch("builtins.open", unittest.mock.mock_open()):
                ai1._handle([mock_file])

            mock_add_to_db.assert_called()

    def test_extraction_signatures(self):
        """Test that extraction functions take correct parameters."""
        # Just check signature or call them with mocks
        with patch("ai1.pdfplumber.open") as mock_pdf_open:
            ai1._extract_pdf("test.pdf")
            mock_pdf_open.assert_called_with("test.pdf")

if __name__ == "__main__":
    unittest.main()
