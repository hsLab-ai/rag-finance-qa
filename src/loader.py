import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pypdf import PdfReader


@dataclass
class LoadedDocument:
    text: str
    source: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    def __init__(self, data_path: str):
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Data directory not found at: {data_path}")
        self.data_path = data_path

    def load_documents(self) -> List[LoadedDocument]:
        print(f"Loading documents from: {self.data_path}")
        loaded_docs: List[LoadedDocument] = []

        for filename in os.listdir(self.data_path):
            if not filename.lower().endswith('.pdf'):
                continue

            filepath = os.path.join(self.data_path, filename)
            print(f"Processing {filename}...")

            try:
                reader = PdfReader(filepath)
                text_parts: List[str] = []

                for page_index, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()

                    if not page_text:
                        continue

                    labeled_page_text = f"[Page {page_index + 1}]\n{page_text}"
                    text_parts.append(labeled_page_text)

                full_text = "\n\n".join(text_parts).strip()

                if not full_text:
                    print(f"Warning: Extracted no text from {filename}. Skipping.")
                    continue

                metadata: Dict[str, Any] = {
                    "num_pages": len(reader.pages),
                    "file_name": filename,
                    "file_path": filepath,
                }

                doc = LoadedDocument(
                    text=full_text,
                    source=filename,
                    path=filepath,
                    metadata=metadata,
                )
                loaded_docs.append(doc)

                print(f"Successfully extracted {len(full_text):,} characters from {filename}.")

            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")

        if not loaded_docs:
            print("Warning: No documents were successfully loaded.")

        return loaded_docs


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DOCS_PATH = os.path.join(BASE_DIR, 'data', 'sample_docs')

    print(f"Calculated Data Path: {SAMPLE_DOCS_PATH}")

    os.makedirs(SAMPLE_DOCS_PATH, exist_ok=True)

    try:
        loader = DocumentLoader(SAMPLE_DOCS_PATH)
        documents = loader.load_documents()

        print("\n--- Document Loading Summary ---")
        for doc in documents:
            print(f"'{doc.source}': {len(doc.text):,} chars, pages={doc.metadata.get('num_pages')}")
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure the directory structure exists and contains PDF files for testing.")
