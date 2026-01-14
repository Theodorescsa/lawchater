# core/ingest.py

import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Xác định đường dẫn (Giữ nguyên format Path của code cũ)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(BASE_DIR / "data")
PERSIST_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "law_docs"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

def main():
    print(" Bắt đầu nạp dữ liệu (Ingest)...")
    
    if not os.path.exists(DATA_PATH):
        print(f" Lỗi: Không tìm thấy folder data tại {DATA_PATH}")
        return

    # 1. Load Documents
    documents = []
    def load_docs(glob, loader_cls):
        loader = DirectoryLoader(DATA_PATH, glob=glob, loader_cls=loader_cls, show_progress=True)
        try: return loader.load()
        except: return []

    documents.extend(load_docs("**/*.pdf", PyPDFLoader))
    documents.extend(load_docs("**/*.docx", Docx2txtLoader))
    
    if not documents:
        print(" Folder data rỗng.")
        return

    # 2. Tiền xử lý (Thêm tên file vào nội dung)
    print(" Đang xử lý metadata...")
    for doc in documents:
        source_file = os.path.basename(doc.metadata.get('source', ''))
        doc.metadata['source_name'] = source_file
        # Gắn tên file vào nội dung để chunk nào cũng biết mình thuộc luật nào
        doc.page_content = f"Tài liệu: {source_file}\n{doc.page_content}"

    # 3. Chia nhỏ (Split) - CẬP NHẬT LOGIC THEO CODE MỚI
    print(" Đang chia nhỏ văn bản...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,   # Giảm xuống 1024 theo code mới (tốt cho E5)
        chunk_overlap=150, # Overlap 150
        separators=[       # Ưu tiên cắt theo "Điều"
            "\n\nĐiều ",
            "\nĐiều ",
            "Điều ",
            "\n\n",
            ". ",
        ],
        keep_separator=True
    )
    chunks = splitter.split_documents(documents)

    # 4. Thêm prefix cho E5 Model
    final_chunks = []
    for chunk in chunks:
        # Bắt buộc cho model intfloat/multilingual-e5-small
        chunk.page_content = f"passage: {chunk.page_content}"
        final_chunks.append(chunk)

    print(f" Số lượng chunks sau khi chia: {len(final_chunks)}")

    # 5. Embedding & Lưu vào ChromaDB
    print(" Đang ghi vào Database...")
    # Cập nhật device='cuda' theo code mới để chạy nhanh hơn (nếu có GPU)
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': 'cuda'} 
    )
    
    # Xóa DB cũ nếu có để làm sạch (FIX LỖI DEVICE BUSY - GIỮ NGUYÊN CODE CŨ)
    if os.path.exists(PERSIST_PATH):
        # Thay vì xóa thư mục (gây lỗi nếu là mount point), ta xóa nội dung bên trong
        for filename in os.listdir(PERSIST_PATH):
            file_path = os.path.join(PERSIST_PATH, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Xóa file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Xóa thư mục con
            except Exception as e:
                print(f" Cảnh báo: Không thể xóa {file_path}. Lỗi: {e}")

    Chroma.from_documents(
        documents=final_chunks, 
        embedding=embedding, 
        collection_name=COLLECTION_NAME, 
        persist_directory=PERSIST_PATH
    )
    
    print(f" Hoàn tất! Đã lưu {len(final_chunks)} chunks vào {PERSIST_PATH}")

if __name__ == "__main__":
    main()