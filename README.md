# RAG Finance QA

A minimal Retrieval-Augmented Generation (RAG) system for financial document question answering.

**PDF → Text → Chunking → Embedding → FAISS Retrieval → LLM Answering**

---

## Features
- Load financial PDFs (annual reports, risk documents, research papers)
- Text chunking with overlap
- Embedding via MiniLM (Sentence-Transformers)
- Vector search using FAISS
- OpenAI GPT-4.1 / GPT-4.1-mini (optional)
- Fallback Dummy LLM (offline mode)
- Simple Streamlit UI for interactive Q&A

---

## Project Structure
```
src/
│
├── loader.py        # PDF → text
├── chunker.py       # text → chunks
├── embedder.py      # chunks → embeddings
├── vector_db.py     # FAISS vector store
├── rag_pipeline.py  # end-to-end RAG pipeline
└── app.py           # Streamlit UI
```

---

## Setup
Install dependencies:

```
pip install -r requirements.txt
```

(Optional) Set your OpenAI key:
```
export OPENAI_API_KEY="your-key"
```

Run the UI:
```
streamlit run src/app.py
```

Put your PDFs in:
```
data/sample_docs/
```
## Usage

Ask questions such as:

- “What is the main theme of this annual report?”
- “What risks does the document highlight?”
- “What are the key financial indicators mentioned?”

The system retrieves relevant text chunks and generates an answer based on the documents.

![RAG Finance QA UI](images/rag_UI.png)


---

## Notes

- If no OpenAI API key is set, the app automatically uses a **DummyLLM** (no real LLM calls are made).
- FAISS index is cached in the `data/` directory and rebuilt on demand.


