from dataclasses import dataclass, field
from typing import List, Dict, Any
from loader import LoadedDocument


@dataclass
class TextChunk:
    text: str
    source: str
    doc_path: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: LoadedDocument) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        text = doc.text
        length = len(text)

        start = 0
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    source=doc.source,
                    doc_path=doc.path,
                    start_idx=start,
                    end_idx=end,
                    metadata={"num_pages": doc.metadata.get("num_pages")},
                )
                chunks.append(chunk)

            if end >= length:
                break

            start = end - self.overlap
            if start < 0:
                start = 0

        return chunks

    def chunk_documents(self, docs: List[LoadedDocument]) -> List[TextChunk]:
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
        return all_chunks


if __name__ == '__main__':
    from loader import DocumentLoader
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, 'data', 'sample_docs')

    loader = DocumentLoader(SAMPLE_DOCS_PATH)
    docs = loader.load_documents()

    chunker = SimpleChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk_documents(docs)

    print("\n--- Chunking Summary ---")
    print(f"Total documents: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Example chunk:\n{chunks[0].text[:300]} ...")
