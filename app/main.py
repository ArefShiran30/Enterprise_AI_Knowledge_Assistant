from pathlib import Path
import json
import shutil

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI(
    title="Enterprise AI Knowledge Assistant"
)


# -----------------------------
# Application folders
# -----------------------------

UPLOAD_FOLDER = Path("uploads")
VECTOR_FOLDER = Path("vector_store")

UPLOAD_FOLDER.mkdir(exist_ok=True)
VECTOR_FOLDER.mkdir(exist_ok=True)


# -----------------------------
# Embedding model
# -----------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# -----------------------------
# Basic endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Enterprise AI Assistant is running"
    }


# -----------------------------
# PDF text extraction
# -----------------------------

def extract_text_from_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from every page of a PDF.
    """

    try:
        reader = PdfReader(file_path)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The PDF could not be opened.",
        ) from error

    extracted_pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        extracted_pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
            }
        )

    return extracted_pages


# -----------------------------
# Chunking
# -----------------------------

def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    step_size = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step_size

    return chunks


def create_document_chunks(
    pages: list[dict],
    filename: str,
) -> list[dict]:
    """
    Create chunks while keeping citation information.
    """

    document_chunks = []
    chunk_id = 1

    for page in pages:
        page_chunks = split_text_into_chunks(
            text=page["text"],
            chunk_size=1000,
            chunk_overlap=200,
        )

        for chunk_text in page_chunks:
            document_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    return document_chunks


# -----------------------------
# Embeddings
# -----------------------------

def create_embeddings(
    chunks: list[dict],
) -> np.ndarray:
    """
    Convert chunk text into numerical embedding vectors.
    """

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the PDF.",
        )

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


# -----------------------------
# FAISS storage
# -----------------------------

def save_vector_store(
    embeddings: np.ndarray,
    chunks: list[dict],
    document_name: str,
) -> dict:
    """
    Store embeddings in FAISS and metadata in JSON.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    vector_dimension = embeddings.shape[1]

    # Because embeddings are normalized, inner-product
    # similarity behaves like cosine similarity.
    index = faiss.IndexFlatIP(vector_dimension)

    index.add(embeddings)

    safe_document_name = Path(document_name).stem

    index_path = (
        VECTOR_FOLDER /
        f"{safe_document_name}.faiss"
    )

    metadata_path = (
        VECTOR_FOLDER /
        f"{safe_document_name}.json"
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            chunks,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "vector_dimension": vector_dimension,
        "vectors_stored": index.ntotal,
    }


# -----------------------------
# Upload endpoint
# -----------------------------

@app.post("/upload")
def upload_pdf(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File has no name.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_FOLDER / safe_filename

    try:
        with file_path.open("wb") as saved_file:
            shutil.copyfileobj(
                file.file,
                saved_file,
            )

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail="The PDF could not be saved.",
        ) from error

    finally:
        file.file.close()

    pages = extract_text_from_pdf(file_path)

    chunks = create_document_chunks(
        pages=pages,
        filename=safe_filename,
    )

    embeddings = create_embeddings(chunks)

    vector_store_information = save_vector_store(
        embeddings=embeddings,
        chunks=chunks,
        document_name=safe_filename,
    )

    return {
        "message": (
            "PDF uploaded, extracted, chunked, "
            "embedded, and stored successfully."
        ),
        "filename": safe_filename,
        "number_of_pages": len(pages),
        "number_of_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_store": vector_store_information,
    }
# -----------------------------
# Search endpoint
# -----------------------------

class SearchRequest(BaseModel):
    document_name: str
    question: str
    top_k: int = 3


@app.post("/search")
def search_document(request: SearchRequest):
    document_name = Path(request.document_name).stem

    index_path = VECTOR_FOLDER / f"{document_name}.faiss"
    metadata_path = VECTOR_FOLDER / f"{document_name}.json"

    if not index_path.exists() or not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document vector store not found. Upload the PDF first.",
        )

    index = faiss.read_index(str(index_path))

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        chunks = json.load(metadata_file)

    question_embedding = embedding_model.encode(
        [request.question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    top_k = min(request.top_k, index.ntotal)

    similarities, indices = index.search(
        question_embedding,
        top_k,
    )

    results = []

    for similarity, chunk_index in zip(
        similarities[0],
        indices[0],
    ):
        if chunk_index == -1:
            continue

        chunk = chunks[int(chunk_index)]

        results.append({
            "similarity": float(similarity),
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
        })

    return {
        "question": request.question,
        "results": results,
    }