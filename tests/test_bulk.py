import sys
import unittest
from unittest.mock import MagicMock, patch
sys.modules["sentence_transformers"] = MagicMock()

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
mock_st.sidebar = MagicMock(__enter__=lambda self: None, __exit__=lambda *args: None)
sys.modules["streamlit"] = mock_st

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

class TestBulkProcessing(unittest.TestCase):
    def setUp(self):
        # Reset session state by clearing the dictionary instead of reassigning
        ai1.st.session_state.clear()
        mock_st.session_state = ai1.st.session_state
        self.mock_faiss_instance = MagicMock()
        ai1.FAISS.return_value = self.mock_faiss_instance
        ai1.HuggingFaceEmbeddings.return_value = MagicMock()
        ai1.faiss.get_num_gpus.return_value = 0

    def test_bulk_processing_with_duplicates(self):
        ai1.init_session()

        file1 = MagicMock()
        file1.name = "doc1.pdf"
        file1.size = 100
        file1.read.side_effect = [b"content A", b""]
        file1.seek = MagicMock()

        file2 = MagicMock()
        file2.name = "doc1_copy.pdf"
        file2.size = 100
        file2.read.side_effect = [b"content A", b""] # Same content => same hash
        file2.seek = MagicMock()

        file3 = MagicMock()
        file3.name = "doc2.pdf"
        file3.size = 200
        file3.read.side_effect = [b"content B", b""]
        file3.seek = MagicMock()

        with patch("ai1._process") as mock_process:
            # We must differentiate between different calls if we want, but returning generic is fine
            mock_process.return_value = (["chunk"], "PDF")

            ai1.add_many_to_db([file1, file2, file3])

            # _process should be called exactly twice because file2 has the same hash as file1
            self.assertEqual(mock_process.call_count, 2)

            # The add_texts should be called exactly once for the batch
            store = mock_st.session_state["vector_store"]
            self.assertEqual(store.add_texts.call_count, 1)

            # Check args given to add_texts
            args, kwargs = store.add_texts.call_args
            self.assertEqual(len(args[0]), 2) # Two chunks from the two distinct files processed

if __name__ == "__main__":
    unittest.main()
