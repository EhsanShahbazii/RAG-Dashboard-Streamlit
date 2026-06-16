import hashlib
import html
import io
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


GAPGPT_CHAT_URL = "https://api.gapgpt.app/v1/chat/completions"
GAPGPT_EMBEDDINGS_URL = "https://api.gapgpt.app/v1/embeddings"
DEFAULT_CHAT_MODEL = "gpt-5-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class Chunk:
    id: int
    page: int
    text: str
    char_count: int
    word_count: int


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.7rem;
            max-width: 1180px;
        }

        :root {
            --rag-border: color-mix(in srgb, currentColor 18%, transparent);
            --rag-muted: color-mix(in srgb, currentColor 66%, transparent);
            --rag-panel: color-mix(in srgb, currentColor 5%, transparent);
            --rag-track: color-mix(in srgb, currentColor 12%, transparent);
            --rag-meter: color-mix(in srgb, currentColor 72%, transparent);
        }

        [data-testid="stSidebar"] {
            background: transparent;
        }

        [data-testid="stSidebar"] * {
            color: inherit;
        }

        .hero {
            border-bottom: 1px solid var(--rag-border);
            margin-bottom: 1.1rem;
            padding-bottom: 1rem;
        }

        .hero h1 {
            font-size: 2.15rem;
            line-height: 1.05;
            margin: 0 0 .45rem 0;
            letter-spacing: 0;
        }

        .hero p {
            color: var(--rag-muted);
            font-size: 1rem;
            margin: 0;
        }

        .stage-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
            margin: .8rem 0 1rem 0;
        }

        .stage {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: .75rem;
            background: var(--rag-panel);
            min-height: 92px;
        }

        .stage small {
            display: block;
            color: var(--rag-muted);
            font-weight: 650;
            margin-bottom: .25rem;
        }

        .stage strong {
            display: block;
            color: inherit;
            font-size: 1.05rem;
            line-height: 1.2;
        }

        .stage span {
            display: block;
            color: var(--rag-muted);
            font-size: .86rem;
            margin-top: .35rem;
        }

        .source-box {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            padding: .85rem;
            background: var(--rag-panel);
            color: inherit;
            margin-bottom: .7rem;
        }

        .source-meta {
            color: var(--rag-muted);
            font-size: .88rem;
            font-weight: 650;
            margin-bottom: .45rem;
        }

        .scorebar {
            height: 8px;
            background: var(--rag-track);
            border-radius: 99px;
            overflow: hidden;
            margin: .35rem 0 .55rem 0;
        }

        .scorebar > div {
            height: 8px;
            background: var(--rag-meter);
            border-radius: 99px;
        }

        @media (max-width: 820px) {
            .stage-row {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 560px) {
            .stage-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_gapgpt_key() -> str:
    try:
        secret_value = st.secrets.get("GAPGPT_API_KEY", "")
    except Exception:
        secret_value = ""
    return secret_value or os.getenv("GAPGPT_API_KEY", "")


def file_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    if PdfReader is None:
        raise RuntimeError("Install pypdf first: python -m pip install pypdf")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": index, "text": text})
    return pages


def split_piece(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]

    sentences = re.split(r"(?<=[.!?\u061f])\s+", piece)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start: start + max_chars].strip())
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())
    return chunks


def chunk_page_text(page_text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(
        r"\n\s*\n", page_text) if item.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            for part in split_piece(paragraph, max_chars):
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(part)
            continue

        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks

    overlapped: list[str] = []
    previous_tail = ""
    for chunk in chunks:
        merged = f"{previous_tail}\n\n{chunk}".strip(
        ) if previous_tail else chunk
        overlapped.append(merged)
        previous_tail = chunk[-overlap_chars:]
    return overlapped


def build_chunks(pages: list[dict[str, Any]], max_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_page_text(page["text"], max_chars, overlap_chars):
            chunks.append(
                Chunk(
                    id=len(chunks) + 1,
                    page=page["page"],
                    text=text,
                    char_count=len(text),
                    word_count=len(text.split()),
                )
            )
    return chunks


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def embed_texts(
    api_key: str,
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    on_batch: Any | None = None,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        response = requests.post(
            GAPGPT_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model_name, "input": batch},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        vectors = [item["embedding"] for item in sorted(
            data["data"], key=lambda item: item["index"])]
        all_vectors.extend(vectors)
        if on_batch:
            on_batch(min(start + len(batch), len(texts)), len(texts))

    return normalize_vectors(np.asarray(all_vectors, dtype=np.float32))


def retrieve(
    api_key: str,
    query: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    query_vector = embed_texts(api_key, [query], model_name)[0]
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk": chunk,
            }
        )
    return results


def build_prompt(question: str, results: list[dict[str, Any]]) -> str:
    context_blocks = []
    for item in results:
        chunk = item["chunk"]
        context_blocks.append(
            f"[Source {item['rank']} | page {chunk.page} | chunk {chunk.id} | score {item['score']:.3f}]\n{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""Use the PDF context below to answer the user's question.

Rules:
- Answer only from the provided context.
- If the answer is not in the context, say you could not find it in the PDF.
- Cite the relevant page and chunk ids in the answer.
- Keep the answer clear and concise.

PDF context:
{context}

Question:
{question}
"""


def ask_gapgpt(api_key: str, model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful PDF research assistant. Ground every answer in retrieved document context.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    response = requests.post(
        GAPGPT_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def render_stage_cards(
    page_count: int,
    chunk_count: int,
    embedding_shape: tuple[int, ...] | None,
    ready: bool,
) -> None:
    vector_text = "Not built yet"
    if embedding_shape:
        vector_text = f"{embedding_shape[0]} vectors x {embedding_shape[1]} dims"

    st.markdown(
        f"""
        <div class="stage-row">
            <div class="stage">
                <small>1. Parse</small>
                <strong>{page_count} pages</strong>
                <span>Extract readable text from the PDF.</span>
            </div>
            <div class="stage">
                <small>2. Split</small>
                <strong>{chunk_count} chunks</strong>
                <span>Make overlapping passages for retrieval.</span>
            </div>
            <div class="stage">
                <small>3. Embed</small>
                <strong>{vector_text}</strong>
                <span>Send chunks to the embedding API.</span>
            </div>
            <div class="stage">
                <small>4. Ask</small>
                <strong>{"Ready" if ready else "Waiting"}</strong>
                <span>Retrieve context, assemble prompt, call the LLM.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("No retrieval results yet.")
        return

    max_score = max((item["score"] for item in results), default=1.0)
    for item in results:
        chunk = item["chunk"]
        percent = 0 if max_score <= 0 else max(
            0, min(100, int((item["score"] / max_score) * 100)))
        snippet = html.escape(chunk.text[:1200]).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="source-box">
                <div class="source-meta">Rank {item['rank']} - similarity {item['score']:.3f} - page {chunk.page} - chunk {chunk.id}</div>
                <div class="scorebar"><div style="width:{percent}%"></div></div>
                <div>{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def reset_document_state() -> None:
    for key in ["pdf_hash", "pages", "chunks", "embeddings", "chat_history", "last_results", "last_prompt"]:
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="VectorRAG PDF Assistant", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>VectorRAG PDF Assistant</h1>
            <p>Upload a PDF, watch the retrieval pipeline form, then ask grounded questions with visible sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pipeline")
        uploaded_file = st.file_uploader("PDF file", type=["pdf"])
        chunk_size = st.slider("Chunk size", 400, 2200, 950, 50)
        overlap = st.slider("Overlap", 0, 500, 160, 25)
        top_k = st.slider("Retrieved chunks", 1, 8, 4)
        chat_model = st.text_input("Chat model", DEFAULT_CHAT_MODEL)
        embedding_model = st.text_input(
            "Embedding model", DEFAULT_EMBEDDING_MODEL)
        embedding_batch_size = st.slider("Embedding batch size", 8, 128, 64, 8)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        api_key = get_gapgpt_key()
        if api_key:
            st.success("GapGPT key found")
        else:
            st.warning("Set GAPGPT_API_KEY before indexing or asking")

        if st.button("Reset PDF", use_container_width=True):
            reset_document_state()
            st.rerun()

    if uploaded_file is None:
        render_stage_cards(0, 0, None, False)
        st.info("Upload a PDF from the sidebar to build the VectorRAG index.")
        return

    pdf_bytes = uploaded_file.getvalue()
    current_hash = file_fingerprint(
        pdf_bytes + f"{chunk_size}:{overlap}:{embedding_model}".encode())

    needs_processing = st.session_state.get("pdf_hash") != current_hash
    if needs_processing:
        reset_document_state()
        st.session_state["pdf_hash"] = current_hash
        with st.status("Parsing and splitting the PDF...", expanded=True) as status:
            st.write("Reading PDF bytes and extracting text page by page.")
            try:
                pages = extract_pages(pdf_bytes)
            except RuntimeError as exc:
                status.update(label="PDF parser is not installed",
                              state="error", expanded=True)
                st.error(str(exc))
                return
            time.sleep(0.15)

            st.write("Splitting pages into overlapping chunks.")
            chunks = build_chunks(pages, chunk_size, overlap)
            time.sleep(0.15)

            if not chunks:
                status.update(label="No readable text found",
                              state="error", expanded=True)
                st.error(
                    "I could not extract readable text from this PDF. It may be scanned image-only.")
                return

            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["embeddings"] = None
            st.session_state["chat_history"] = []
            st.session_state["last_results"] = []
            st.session_state["last_prompt"] = ""
            status.update(label="PDF parsed and split",
                          state="complete", expanded=False)

    pages = st.session_state.get("pages", [])
    chunks = st.session_state.get("chunks", [])
    embeddings = st.session_state.get("embeddings")

    if chunks and embeddings is None and api_key:
        with st.status("Embedding chunks through GapGPT API...", expanded=True) as status:
            st.write(f"Sending {len(chunks)} chunks to {embedding_model}.")

            def report_embedding_progress(done: int, total: int) -> None:
                st.write(f"Embedded {done} of {total} chunks.")

            try:
                embeddings = embed_texts(
                    api_key,
                    [chunk.text for chunk in chunks],
                    embedding_model,
                    batch_size=embedding_batch_size,
                    on_batch=report_embedding_progress,
                )
            except requests.HTTPError as exc:
                detail = exc.response.text[:800] if exc.response is not None else str(
                    exc)
                status.update(label="Embedding request failed",
                              state="error", expanded=True)
                st.error(f"GapGPT embeddings request failed: {detail}")
            except requests.RequestException as exc:
                status.update(label="Embedding request failed",
                              state="error", expanded=True)
                st.error(f"Could not reach GapGPT embeddings API: {exc}")
            else:
                st.session_state["embeddings"] = embeddings
                status.update(label="Vector index is ready",
                              state="complete", expanded=False)

    embeddings = st.session_state.get("embeddings")

    render_stage_cards(
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_shape=None if embeddings is None else embeddings.shape,
        ready=bool(chunks) and embeddings is not None,
    )

    ask_tab, chunks_tab, retrieval_tab, prompt_tab = st.tabs(
        ["Ask", "Splits", "Retrieval", "Prompt"]
    )

    with ask_tab:
        for message in st.session_state.get("chat_history", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the uploaded PDF")
        if question:
            if not api_key:
                st.error(
                    "Please set GAPGPT_API_KEY in your environment or Streamlit secrets.")
                return
            if embeddings is None:
                st.error(
                    "The PDF has been split, but the vector index is not ready yet.")
                return

            st.session_state["chat_history"].append(
                {"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.status("Answering with VectorRAG...", expanded=True) as status:
                    st.write("Embedding your question through GapGPT.")
                    try:
                        results = retrieve(
                            api_key, question, chunks, embeddings, embedding_model, top_k)
                    except requests.HTTPError as exc:
                        detail = exc.response.text[:800] if exc.response is not None else str(
                            exc)
                        status.update(label="Question embedding failed",
                                      state="error", expanded=True)
                        st.error(f"GapGPT embeddings request failed: {detail}")
                        return
                    except requests.RequestException as exc:
                        status.update(label="Question embedding failed",
                                      state="error", expanded=True)
                        st.error(
                            f"Could not reach GapGPT embeddings API: {exc}")
                        return

                    st.write(
                        f"Selecting the top {len(results)} most similar chunks.")
                    prompt = build_prompt(question, results)

                    st.write("Creating the grounded prompt for the LLM.")
                    st.session_state["last_results"] = results
                    st.session_state["last_prompt"] = prompt

                    st.write(f"Calling {chat_model} through GapGPT.")
                    try:
                        answer = ask_gapgpt(
                            api_key, chat_model, prompt, temperature)
                    except requests.HTTPError as exc:
                        detail = exc.response.text[:800] if exc.response is not None else str(
                            exc)
                        status.update(label="LLM request failed",
                                      state="error", expanded=True)
                        st.error(f"GapGPT request failed: {detail}")
                        return
                    except requests.RequestException as exc:
                        status.update(label="LLM request failed",
                                      state="error", expanded=True)
                        st.error(f"Could not reach GapGPT: {exc}")
                        return
                    status.update(label="Answer ready",
                                  state="complete", expanded=False)

                st.markdown(answer)
                with st.expander("Sources used for this answer", expanded=False):
                    render_sources(st.session_state["last_results"])

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer})

    with chunks_tab:
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            page_filter = st.selectbox(
                "Page", ["All"] + sorted({chunk.page for chunk in chunks}))
        with col_b:
            chunk_limit = st.slider("Visible chunks", 5, min(
                80, max(5, len(chunks))), min(20, max(5, len(chunks))))
        with col_c:
            split_search = st.text_input("Search splits")

        visible_chunks = chunks
        if page_filter != "All":
            visible_chunks = [
                chunk for chunk in visible_chunks if chunk.page == page_filter]
        if split_search:
            needle = split_search.lower()
            visible_chunks = [
                chunk for chunk in visible_chunks if needle in chunk.text.lower()]

        for chunk in visible_chunks[:chunk_limit]:
            with st.expander(
                f"Chunk {chunk.id} - page {chunk.page} - {chunk.word_count} words - {chunk.char_count} chars",
                expanded=False,
            ):
                st.write(chunk.text)

    with retrieval_tab:
        st.subheader("Latest Retrieval")
        render_sources(st.session_state.get("last_results", []))

    with prompt_tab:
        st.subheader("Latest Prompt")
        prompt_text = st.session_state.get("last_prompt", "")
        if prompt_text:
            st.code(prompt_text, language="markdown")
        else:
            st.info("Ask a question to see the exact context prompt sent to the LLM.")


if __name__ == "__main__":
    main()
