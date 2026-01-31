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
        """Test that init_session initializes vector_store in session_state and DOES NOT load from disk."""
        # Ensure _load_session is NOT called (it shouldn't exist)
        self.assertFalse(hasattr(ai1, "_load_session"), "Global _load_session should be removed")
        self.assertFalse(hasattr(ai1, "SESSION_FILE"), "Global SESSION_FILE should be removed")

        ai1.init_session()

        self.assertIn("vector_store", mock_st.session_state)
        self.assertIsNotNone(mock_st.session_state["vector_store"])

    def test_handle_upload(self):
        """Test that _handle processes upload without saving to disk."""
        # Setup session state
        ai1.init_session()

        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.size = 1234
        # Simulate chunked reading: return data then empty bytes to signal EOF
        mock_file.read.side_effect = [b"fake content", b""]
        mock_file.seek = MagicMock()

        # Mock _process to return some chunks
        with patch("ai1._process") as mock_process:
            mock_process.return_value = (["chunk1", "chunk2"], "PDF")

            # Call _handle
            ai1._handle([mock_file])

            # Verify file was read and reset
            mock_file.read.assert_called()
            # It should be called at least twice (content + EOF)
            self.assertGreaterEqual(mock_file.read.call_count, 2)
            mock_file.seek.assert_called_with(0)

            # Verify _process was called with file object
            mock_process.assert_called()
            args, _ = mock_process.call_args
            self.assertEqual(args[0], mock_file) # Should pass the file object

            # Verify added to vector store
            # The vector store in session state should have add_texts called
            store = mock_st.session_state["vector_store"]
            # ai1.FAISS() returns the mock_faiss_instance?
            # When init_session runs, it calls _new_store() -> FAISS(...)
            # So st.session_state['vector_store'] is an instance of the mock FAISS.

            # Note: since we patched ai1.FAISS globally in setUp, new instances are mocks.
            # But init_session calls _new_store() which calls FAISS().
            # So store should be a mock.
            store.add_texts.assert_called()

            # Verify NO file was opened/written
            self.assertFalse(hasattr(ai1, "_save_session"))

    def test_extraction_signatures(self):
        """Test that extraction functions take file objects."""
        # Just check signature or call them with mocks
        with patch("ai1.pdfplumber.open") as mock_pdf_open:
            fobj = MagicMock()
            ai1._extract_pdf(fobj, "test.pdf")
            mock_pdf_open.assert_called_with(fobj)

    def test_handle_upload_chunked(self):
        """Test that _handle reads file in chunks to prevent DoS."""
        ai1.init_session()
        mock_file = MagicMock()
        mock_file.name = "large.pdf"
        mock_file.size = 100000
        # return 2 chunks then EOF
        mock_file.read.side_effect = [b"A" * 65536, b"B" * 34464, b""]
        mock_file.seek = MagicMock()

        with patch("ai1._process") as mock_process:
            mock_process.return_value = ([], "PDF")
            ai1._handle([mock_file])

            # Check calls to read
            # Must be called with an integer argument (chunk size)
            calls = mock_file.read.mock_calls
            for c in calls:
                # mock_calls entries are Call objects, which can be unpacked as (name, args, kwargs)
                # or just use .args
                if not c.args:
                   # It might be __enter__ or similar if used as context manager,
                   # but here read() is simple method.
                   # Actually, if read() is called without args, args will be empty tuple.
                   pass

                # If it's a call to read, check args
                # We expect read(65536)
                if c.args:
                     self.assertTrue(isinstance(c.args[0], int), "read() called without size limit")
                     self.assertEqual(c.args[0], 65536)
                else:
                     # call without args?
                     self.fail("read() called without arguments")

if __name__ == "__main__":
    unittest.main()
