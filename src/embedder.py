from dataclasses import dataclass, field
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from chunker import TextChunk, SimpleChunker
from loader import DocumentLoader


@dataclass
class EmbeddedChunk:
    vector: List[float]
    text: str
    source: str
    doc_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        if not chunks:
            print("No chunks provided for embedding.")
            return []

        texts = [chunk.text for chunk in chunks]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        embedded_chunks: List[EmbeddedChunk] = []
        for chunk, vec in zip(chunks, embeddings):
            embedded = EmbeddedChunk(
                vector=vec.tolist(),
                text=chunk.text,
                source=chunk.source,
                doc_path=chunk.doc_path,
                metadata={
                    **chunk.metadata,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                },
            )
            embedded_chunks.append(embedded)

        print(f"Finished embedding {len(embedded_chunks)} chunks.")
        return embedded_chunks


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, "data", "sample_docs")

    print(f"Sample docs path: {SAMPLE_DOCS_PATH}")

    loader = DocumentLoader(SAMPLE_DOCS_PATH)
    docs = loader.load_documents()

    if not docs:
        print("No documents loaded, cannot embed.")
        raise SystemExit(0)

    first_doc = docs[0]
    print(f"Using first document: {first_doc.source}")

    chunker = SimpleChunker(chunk_size=1000, overlap=200)
    chunk
