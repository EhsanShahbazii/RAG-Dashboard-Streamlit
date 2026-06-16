# VectorRAG PDF Assistant

A Streamlit app for uploading a PDF, inspecting the VectorRAG pipeline, and asking grounded questions with visible retrieval sources.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GAPGPT_API_KEY="your-key-here"
streamlit run app.py
```

You can also use Streamlit secrets:

```toml
# .streamlit/secrets.toml
GAPGPT_API_KEY = "your-key-here"
```

## What the app shows

- PDF text extraction by page
- Overlapping chunk splits
- GapGPT embedding vector shape
- Retrieval scores and source chunks
- The exact prompt sent to the LLM
- Chat answers with page and chunk references

GapGPT is used for both embeddings and generation. During indexing, the app sends PDF chunks to `/v1/embeddings`; during question answering, it sends only the retrieved context to `/v1/chat/completions`.
