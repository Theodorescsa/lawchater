# core/check_db.py
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from django.conf import settings
from pathlib import Path

# Setup path giống hệt app.py
BASE_DIR = Path(__file__).resolve().parent
PERSIST_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "law_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

print(f"📂 Checking DB at: {PERSIST_PATH}")

try:
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_PATH,
        embedding_function=embedding
    )
    
    count = vectorstore._collection.count()
    print(f"✅ KẾT QUẢ: Trong DB hiện có {count} chunks.")
    
    if count == 0:
        print("⚠️ CẢNH BÁO: DB rỗng! Bạn cần vào Admin Django -> Chọn tài liệu -> Action 'Ingest selected documents' lại.")
        
except Exception as e:
    print(f"❌ LỖI: {e}")