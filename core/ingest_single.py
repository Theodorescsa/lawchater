import os
from django.utils import timezone
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

from app_document.models import UploadedDocument

from django.conf import settings

PERSIST_PATH = settings.BASE_DIR / "core" / "chroma_db"
COLLECTION_NAME = "law_docs"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


def ingest_one_document(doc: UploadedDocument):
    """
    Ingest duy nhất 1 UploadedDocument
    """
    try:
        doc.status = "processing"
        doc.error_message = ""
        doc.save(update_fields=["status", "error_message"])

        file_path = doc.file.path
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        documents = loader.load()

        for d in documents:
            d.metadata.update({
                "document_id": doc.id,
                "user_id": doc.user_id,
                "source_name": doc.original_name,
            })
            d.page_content = f"passage: Tài liệu: {doc.original_name}\n{d.page_content}"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\nĐiều ", "\nĐiều ", "Điều "],
            keep_separator=True,
        )

        chunks = splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )

        db = Chroma(
            persist_directory=str(PERSIST_PATH),
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
        )

        db.add_documents(chunks)

        doc.status = "done"
        doc.processed_at = timezone.now()
        doc.save(update_fields=["status", "processed_at"])

    except Exception as e:
        doc.status = "failed"
        doc.error_message = str(e)
        doc.save(update_fields=["status", "error_message"])
        raise
