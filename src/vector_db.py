from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import pickle

import numpy as np
import faiss

from embedder import EmbeddedChunk, ChunkEmbedder
from chunker import SimpleChunker
from loader import DocumentLoader


@dataclass
class RetrievedChunk:
    score: float
    text: str
    source: str
    doc_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaissVectorStore:
    def __init__(
        self,
        dim: int,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        use_cosine: bool = True,
    ):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.use_cosine = use_cosine

        self.index: Optional[faiss.Index] = None
        self.embedded_chunks: List[EmbeddedChunk] = []

        if index_path and os.path.exists(index_path) and metadata_path and os.path.exists(metadata_path):
            self._load_from_disk()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        if self.use_cosine:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def _load_from_disk(self):
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        print(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.embedded_chunks = pickle.load(f)

        if len(self.embedded_chunks) != self.index.ntotal:
            print("Warning: metadata size does not match index size.")

    def add_embeddings(self, embedded_chunks: List[EmbeddedChunk]):
        if not embedded_chunks:
            return

        vectors = np.array([ec.vector for ec in embedded_chunks], dtype="float32")

        if self.use_cosine:
            faiss.normalize_L2(vectors)

        if self.index is None:
            self._create_empty_index()

        self.index.add(vectors)
        self.embedded_chunks.extend(embedded_chunks)

        print(f"Added {len(embedded_chunks)} vectors to index. Total = {self.index.ntotal}")

    def save(self):
        if not self.index_path or not self.metadata_path:
            raise ValueError("index_path and metadata_path must be set to save the index.")

        print(f"Saving FAISS index to: {self.index_path}")
        faiss.write_index(self.index, self.index_path)

        print(f"Saving metadata to: {self.metadata_path}")
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.embedded_chunks, f)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[RetrievedChunk]:
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty, cannot search.")
            return []

        q = np.array([query_vector], dtype="float32")

        if self.use_cosine:
            faiss.normalize_L2(q)

        distances, indices = self.index.search(q, top_k)

        results: List[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            ec = self.embedded_chunks[idx]
            score = float(dist)

            result = RetrievedChunk(
                score=score,
                text=ec.text,
                source=ec.source,
                doc_path=ec.doc_path,
                metadata=ec.metadata,
            )
            results.append(result)

        return results


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, "data", "sample_docs")
    INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
    META_PATH = os.path.join(BASE_DIR, "data", "faiss_metadata.pkl")

    print(f"Sample docs path: {SAMPLE_DOCS_PATH}")

    loader = DocumentLoader(SAMPLE_DOCS_PATH)
    docs = loader.load_documents()

    if not docs:
        print("No documents loaded, cannot build index.")
        raise SystemExit(0)

    first_doc = docs[0]
    print(f"Using first document: {first_doc.source}")

    chunker = SimpleChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk_document(first_doc)
    print(f"Total chunks from {first_doc.source}: {len(chunks)}")

    if not chunks:
        print("No chunks to embed, abort.")
        raise SystemExit(0)

    chunks_to_use = chunks[:200]

    embedder = ChunkEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks_to_use)

    if not embedded_chunks:
        print("Embedding failed or returned no vectors.")
        raise SystemExit(0)

    dim = len(embedded_chunks[0].vector)
    print(f"Vector dimension: {dim}")

    vector_store = FaissVectorStore(
        dim=dim,
        index_path=INDEX_PATH,
        metadata_path=META_PATH,
        use_cosine=True,
    )

    vector_store.add_embeddings(embedded_chunks)
    vector_store.save()

    query_text = embedded_chunks[0].text[:300]
    print("\n=== Query Text Preview ===")
    print(query_text, "...\n")

    query_vec = embedder.model.encode([query_text], convert_to_numpy=True)[0]
    results = vector_store.search(query_vec, top_k=5)

    print("=== Top-5 Retrieved Chunks ===")
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] score={r.score:.4f}, source={r.source}")
        print(r.text[:200], "...")
