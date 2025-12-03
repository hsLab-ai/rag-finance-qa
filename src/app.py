import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

from loader import DocumentLoader
from chunker import SimpleChunker
from embedder import ChunkEmbedder, EmbeddedChunk
from vector_db import FaissVectorStore
from rag_pipeline import RagPipeline, DummyLLM

try:
    from rag_pipeline import OpenAILLM
except ImportError:
    OpenAILLM = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, "data", "sample_docs")
INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
META_PATH = os.path.join(BASE_DIR, "data", "faiss_metadata.pkl")


def build_pipeline(force_rebuild: bool = False) -> RagPipeline:
    st.info("Loading embedding model (MiniLM). Please wait...")
    embedder = ChunkEmbedder()
    vector_dim = embedder.model.get_sentence_embedding_dimension()

    has_existing_index = os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)

    if has_existing_index and not force_rebuild:
        st.success("Existing FAISS index found. Loading index from disk...")
        vector_store = FaissVectorStore(
            dim=vector_dim,
            index_path=INDEX_PATH,
            metadata_path=META_PATH,
            use_cosine=True,
        )
    else:
        if has_existing_index and force_rebuild:
            st.warning("Rebuilding FAISS index... Existing index will be overwritten.")
        else:
            st.warning("No FAISS index found. Building index from PDF documents...")

        loader = DocumentLoader(SAMPLE_DOCS_PATH)
        docs = loader.load_documents()
        if not docs:
            st.error("No PDF files found in data/sample_docs. Please add documents and retry.")
            st.stop()

        chunker = SimpleChunker(chunk_size=1000, overlap=200)
        all_chunks = []
        for doc in docs:
            doc_chunks = chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        st.write(f"Total chunks from documents: {len(all_chunks)}")

        N = min(2000, len(all_chunks))
        chunks_to_use = all_chunks[:N]
        st.write(f"Using first {N} chunks to build the FAISS index.")

        embedded_chunks: list[EmbeddedChunk] = embedder.embed_chunks(chunks_to_use)

        vector_store = FaissVectorStore(
            dim=vector_dim,
            index_path=INDEX_PATH,
            metadata_path=META_PATH,
            use_cosine=True,
        )
        vector_store.add_embeddings(embedded_chunks)
        vector_store.save()
        st.success("FAISS index successfully built and saved.")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAILLM is not None:
        st.success("OPENAI_API_KEY detected. Using OpenAI model.")
        llm = OpenAILLM(model="gpt-4.1-mini", api_key=api_key)
    else:
        if not api_key:
            st.warning("OPENAI_API_KEY not found — using DummyLLM. (Responses will NOT be real LLM answers.)")
        elif OpenAILLM is None:
            st.warning("OpenAILLM class not found — using DummyLLM.")
        llm = DummyLLM()

    pipeline = RagPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        top_k=5,
    )
    return pipeline


def main():
    st.set_page_config(
        page_title="RAG Finance QA",
        page_icon="💹",
        layout="wide",
    )

    st.title("💹 RAG Finance QA — Financial Document Question Answering")
    st.markdown(
        "Place financial PDF documents inside the `data/sample_docs/` directory. "
        "Then ask questions based on the document content."
        )

    st.sidebar.header("⚙️ Settings")

    force_rebuild = st.sidebar.button("🔁 Rebuild FAISS Index")

    if "pipeline" not in st.session_state or force_rebuild:
        with st.spinner("Building / loading RAG pipeline..."):
            st.session_state.pipeline = build_pipeline(force_rebuild=force_rebuild)

    pipeline: RagPipeline = st.session_state.pipeline

    st.markdown("---")

    st.subheader("📝 Ask a Question")

    default_question = "What is the main focus or theme of this annual report?"
    question = st.text_area(
        "Enter your question below:",
        value=default_question,
        height=80,
    )

    ask_button = st.button("🚀 Ask")

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Running RAG pipeline (embedding → retrieval → LLM)..."):
            result = pipeline.answer(question)

        st.markdown("## ✅ Answer")
        st.write(result.answer)

        st.markdown("## 📚 Retrieved Chunks")
        if not result.retrieved_chunks:
            st.info("No relevant document sections were retrieved.")
        else:
            for i, rc in enumerate(result.retrieved_chunks, start=1):
                with st.expander(f"[{i}] score={rc.score:.3f} | file={rc.source}"):
                    st.write(rc.text)
                    if rc.metadata:
                        st.caption(f"Metadata: {rc.metadata}")


if __name__ == "__main__":
    main()
