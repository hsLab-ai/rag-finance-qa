from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os
import numpy as np

from loader import DocumentLoader
from chunker import SimpleChunker
from embedder import ChunkEmbedder, EmbeddedChunk
from vector_db import FaissVectorStore, RetrievedChunk

from openai import OpenAI


@dataclass
class QAResult:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]


class DummyLLM:
    def generate(self, question: str, context: str) -> str:
        return f"Dummy response.\n\nQuestion:\n{question}\n\nContext:\n{context[:800]}"


class OpenAILLM:
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        self.model = model

    def generate(self, question: str, context: str) -> str:
        user_prompt = (
            "You are a financial document QA assistant. "
            "Answer the question only based on the context below. "
            "If the context is insufficient, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer clearly and concisely:\n"
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant specialized in financial documents.",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.output[0].content[0].text


class RagPipeline:
    def __init__(
        self,
        embedder: ChunkEmbedder,
        vector_store: FaissVectorStore,
        llm,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def _embed_question(self, question: str) -> np.ndarray:
        vec = self.embedder.model.encode([question], convert_to_numpy=True)[0]
        return vec

    def _build_context(self, retrieved: List[RetrievedChunk]) -> str:
        parts = []
        for i, r in enumerate(retrieved, start=1):
            header = f"[Source {i} | score={r.score:.3f} | file={r.source}]"
            body = r.text
            parts.append(header + "\n" + body)
        return "\n\n-----\n\n".join(parts)

    def answer(self, question: str) -> QAResult:
        query_vec = self._embed_question(question)
        retrieved = self.vector_store.search(query_vec.tolist(), top_k=self.top_k)

        if not retrieved:
            return QAResult(
                question=question,
                answer="No relevant content retrieved from the vector store.",
                retrieved_chunks=[],
            )

        context = self._build_context(retrieved)
        answer = self.llm.generate(question=question, context=context)

        return QAResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved,
        )


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, "data", "sample_docs")
    INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
    META_PATH = os.path.join(BASE_DIR, "data", "faiss_metadata.pkl")

    embedder = ChunkEmbedder()
    vector_dim = embedder.model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {vector_dim}")

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading existing FAISS index...")
        vector_store = FaissVectorStore(
            dim=vector_dim,
            index_path=INDEX_PATH,
            metadata_path=META_PATH,
            use_cosine=True,
        )
    else:
        print("Building FAISS index...")

        loader = DocumentLoader(SAMPLE_DOCS_PATH)
        docs = loader.load_documents()
        if not docs:
            print("No documents loaded.")
            raise SystemExit(0)

        chunker = SimpleChunker(chunk_size=1000, overlap=200)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker.chunk_document(doc))

        print(f"Total chunks: {len(all_chunks)}")
        N = min(1000, len(all_chunks))
        chunks_to_use = all_chunks[:N]

        embedded_chunks: List[EmbeddedChunk] = embedder.embed_chunks(chunks_to_use)

        vector_store = FaissVectorStore(
            dim=vector_dim,
            index_path=INDEX_PATH,
            metadata_path=META_PATH,
            use_cosine=True,
        )
        vector_store.add_embeddings(embedded_chunks)
        vector_store.save()

    llm = OpenAILLM(model="gpt-4.1-mini")

    pipeline = RagPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        top_k=5,
    )

    question = "What is the main focus or theme of this annual report?"
    print(f"\n=== Question ===\n{question}\n")

    result = pipeline.answer(question)

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Retrieved Sources ===")
    for i, rc in enumerate(result.retrieved_chunks, start=1):
        print(f"\n[{i}] score={rc.score:.3f}, source={rc.source}")
        print(rc.text[:300], "...")
