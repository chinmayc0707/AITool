import sys
from unittest.mock import MagicMock

# Mock dependencies before importing ai1
mock_st = MagicMock()
sys.modules["streamlit"] = mock_st
sys.modules["requests"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["faiss"] = MagicMock()
sys.modules["pdfplumber"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["langchain"] = MagicMock()
sys.modules["langchain.text_splitter"] = MagicMock()
sys.modules["langchain.embeddings"] = MagicMock()
sys.modules["langchain.vectorstores"] = MagicMock()
sys.modules["langchain.docstore"] = MagicMock()

import ai1
import time
import os

def run_verification():
    # Setup mock PDF
    page = MagicMock()
    page.extract_text.return_value = "Page text"
    page.extract_tables.return_value = [[["cell1", "cell2"], ["cell3", "cell4"]]]

    pdf_mock = MagicMock()
    pdf_mock.pages = [page]

    ai1.pdfplumber.open.return_value.__enter__.return_value = pdf_mock

    # Get output from current implementation
    output = ai1._extract_pdf("dummy.pdf")

    expected_output = "Page text\ncell1 | cell2\ncell3 | cell4"
    if output == expected_output:
        print("Functional Verification: PASSED")
    else:
        print("Functional Verification: FAILED")
        print(f"Expected:\n{repr(expected_output)}")
        print(f"Got:\n{repr(output)}")

if __name__ == "__main__":
    run_verification()
