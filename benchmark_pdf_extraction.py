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

def run_benchmark():
    # Simulate a PDF with 1000 pages, each with some text and a table
    num_pages = 1000
    pages = []
    for i in range(num_pages):
        page = MagicMock()
        page.extract_text.return_value = "This is some sample text for page " + str(i) + ". " * 50
        page.extract_tables.return_value = [[["cell " + str(i) + "," + str(j) + "," + str(k) for k in range(5)] for j in range(10)]]
        pages.append(page)

    pdf_mock = MagicMock()
    pdf_mock.pages = pages

    # We need to mock pdfplumber.open to return our pdf_mock
    ai1.pdfplumber.open.return_value.__enter__.return_value = pdf_mock

    # Warm up
    ai1._extract_pdf("dummy.pdf")

    start = time.perf_counter()
    num_runs = 20
    for _ in range(num_runs):
        ai1._extract_pdf("dummy.pdf")
    avg_time = (time.perf_counter() - start) / num_runs
    print(f"Baseline implementation avg time: {avg_time:.6f}s")

if __name__ == "__main__":
    run_benchmark()
