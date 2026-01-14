# core/ingest_single.py
import os
import torch
from django.utils import timezone
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from django.conf import settings
from app_document.models import UploadedDocument

# --- CẤU HÌNH (Đã đồng bộ với ingest.py) ---
PERSIST_PATH = settings.BASE_DIR / "core" / "chroma_db"
COLLECTION_NAME = "law_docs"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

def ingest_one_document(doc: UploadedDocument):
    """
    Hàm này được gọi khi bấm 'Ingest' trên Django Admin.
    Code đã được đồng bộ logic với ingest.py thủ công.
    """
    try:
        # 1. Cập nhật trạng thái đang xử lý
        doc.status = "processing"
        doc.error_message = ""
        doc.save(update_fields=["status", "error_message"])

        file_path = doc.file.path
        ext = os.path.splitext(file_path)[1].lower()

        # 2. Load file
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        documents = loader.load()

        # 3. Xử lý Metadata & Nội dung (Quan trọng cho Filter)
        for d in documents:
            d.metadata.update({
                "document_id": doc.id,
                "user_id": doc.user_id,
                # [QUAN TRỌNG] Tên file này phải khớp với bộ lọc trong app.py
                "source_name": doc.original_name, 
            })
            # Thêm ngữ cảnh tên file vào nội dung
            d.page_content = f"Tài liệu: {doc.original_name}\n{d.page_content}"

        # 4. Chia nhỏ (Logic mới giống ingest.py)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,   # Đã giảm xuống 1024
            chunk_overlap=150, # Overlap 150
            separators=[       # Ưu tiên cắt theo điều luật
                "\n\nĐiều ",
                "\nĐiều ",
                "Điều ",
                "\n\n",
                ". ",
            ],
            keep_separator=True
        )
        chunks = splitter.split_documents(documents)

        # 5. Thêm prefix 'passage:' (BẮT BUỘC cho model E5)
        final_chunks = []
        for chunk in chunks:
            chunk.page_content = f"passage: {chunk.page_content}"
            final_chunks.append(chunk)

        # 6. Embedding (Tự động chọn GPU/CPU an toàn)
        if torch.cuda.is_available():
            device = "cuda"
            print(f"--> Ingest Admin dùng GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("--> Ingest Admin dùng CPU")
            
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device}
        )

        db = Chroma(
            persist_directory=str(PERSIST_PATH),
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
        )

        # Lưu vào DB
        db.add_documents(final_chunks)

        # 7. Hoàn tất
        doc.status = "done"
        doc.processed_at = timezone.now()
        doc.save(update_fields=["status", "processed_at"])

    except Exception as e:
        print(f"Lỗi Ingest Admin: {e}")
        doc.status = "failed"
        doc.error_message = str(e)
        doc.save(update_fields=["status", "error_message"])
        raise