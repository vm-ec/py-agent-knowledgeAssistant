from openai import OpenAI
from dotenv import load_dotenv
from pinecone_client import get_index
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import fitz
import os
from docx import Document
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100


def load_documents(file_path):
    if file_path.endswith(".pdf"):
        documents = []
        try:
            pdf = fitz.open(file_path)
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": page_number,
                        },
                    })
            pdf.close()
        except Exception:
            documents = []

        if documents:
            return documents

        try:
            reader = PdfReader(file_path, strict=False)
        except PdfReadError as exc:
            raise ValueError(
                "This PDF appears to be malformed and could not be read. "
                "Try opening it in a PDF viewer and printing/saving it as a new PDF, "
                "or upload a text version of the document."
            ) from exc

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            text = text.strip()
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": page_number,
                    },
                })

        if not documents:
            raise ValueError(
                "No readable text was found in this PDF. If it is scanned or image-only, "
                "convert it with OCR first or upload a TXT file."
            )

        return documents
    
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if not text.strip():
            raise ValueError("No readable text found in the Word document.")
        return [{"text": text, "metadata": {"source": os.path.basename(file_path)}}]
    
    elif file_path.endswith(".csv"):
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("CSV file is empty.")
        text = "\n".join([", ".join(row) for row in rows])
        return [{"text": text, "metadata": {"source": os.path.basename(file_path)}}]

    with open(file_path, "r", encoding="utf-8") as f:
        return [{
            "text": f.read(),
            "metadata": {"source": os.path.basename(file_path)},
        }]


def split_documents(documents):
    chunks = []

    for document in documents:
        text = document["text"]
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": document["metadata"],
                })

            if end == len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks

def batch_items(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

def ingest_document(file_path):
    docs = load_documents(file_path)
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks were created from this document.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    index = get_index()

    vectors = []
    for batch_start, chunk_batch in enumerate(batch_items(chunks, EMBED_BATCH_SIZE)):
        response = client.embeddings.create(
            input=[chunk["text"] for chunk in chunk_batch],
            model="text-embedding-3-small"
        )
        for offset, embedding_item in enumerate(response.data):
            chunk_index = batch_start * EMBED_BATCH_SIZE + offset
            chunk = chunk_batch[offset]
            vectors.append({
                "id": f"{os.path.basename(file_path)}_{chunk_index}",
                "values": embedding_item.embedding,
                "metadata": {
                    "text": chunk["text"],
                    **chunk["metadata"],
                },
            })

    for vector_batch in batch_items(vectors, UPSERT_BATCH_SIZE):
        index.upsert(vectors=vector_batch)

    return len(vectors)
