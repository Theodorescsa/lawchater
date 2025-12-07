import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Lấy đường dẫn tuyệt đối của thư mục core
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = str(BASE_DIR / "data")
PERSIST_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "law_docs"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

def main():
    print("Bắt đầu quá trình nạp dữ liệu...")
    print(f"Đường dẫn data: {DATA_PATH}")
    print(f"Đường dẫn persist: {PERSIST_PATH}")
    
    # Kiểm tra thư mục data có tồn tại không
    if not os.path.exists(DATA_PATH):
        print(f"❌ LỖI: Thư mục '{DATA_PATH}' không tồn tại!")
        return
    
    # Liệt kê các file trong thư mục data
    print(f"\nCác file trong thư mục data:")
    for file in os.listdir(DATA_PATH):
        print(f"  - {file}")
    print()

    # --- Tải tài liệu ---
    documents = []
    
    def load_docs(glob_pattern, loader_cls, label):
        print(f"Đang tải file {label} từ '{DATA_PATH}'...")
        loader = DirectoryLoader(
            DATA_PATH,
            glob=glob_pattern,
            loader_cls=loader_cls,
            use_multithreading=True,
            show_progress=True
        )
        try:
            docs = loader.load()
            print(f"✅ Đã tải {len(docs)} tài liệu {label}")
            return docs
        except Exception as e:
            print(f"⚠️ Lỗi khi tải {label}: {e}")
            return []

    documents.extend(load_docs("**/*.pdf", PyPDFLoader, "PDF"))
    documents.extend(load_docs("**/*.docx", Docx2txtLoader, "DOCX"))
    documents.extend(load_docs("**/*.doc", Docx2txtLoader, "DOC"))

    if not documents:
        print(f"❌ Không tìm thấy tài liệu nào trong thư mục '{DATA_PATH}'.")
        print("Vui lòng kiểm tra lại đường dẫn và các file.")
        return

    print(f"\n✅ Đã tải thành công {len(documents)} tài liệu.")
    print(f"Tổng số ký tự: {sum(len(doc.page_content) for doc in documents)}")

    print("\nĐang chia tài liệu thành các mảnh theo 'Điều'...")
    
    logical_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\nĐiều ", "\nĐiều ", "Điều "],
        keep_separator=True
    )
    
    chunks_with_preamble = logical_splitter.split_documents(documents)
    
    final_chunks = []
    for chunk in chunks_with_preamble:
        content = chunk.page_content.lstrip()
        if content.startswith("Điều "):
            chunk.page_content = content
            final_chunks.append(chunk)

    if not final_chunks:
        print("⚠️ CẢNH BÁO: Không tìm thấy 'Điều ' nào trong văn bản.")
        print("Sử dụng tất cả các chunks...")
        final_chunks = chunks_with_preamble

    print(f"✅ Đã chia thành {len(final_chunks)} mảnh logic.")

    print(f"\nĐang tải mô hình embedding '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )
    print("✅ Đã tải embedding model")

    print(f"\nĐang ghi dữ liệu vào ChromaDB (collection: {COLLECTION_NAME})...")
    print("⏳ Quá trình này có thể mất vài phút...")

    try:
        vectorstore = Chroma.from_documents(
            documents=final_chunks,            
            embedding=embedding_model,          
            collection_name=COLLECTION_NAME,    
            persist_directory=PERSIST_PATH    
        )
        print("✅ Đã ghi dữ liệu vào ChromaDB")
    except Exception as e:
        print(f"❌ Lỗi khi ghi vào ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print(f"🎉 Hoàn tất! Dữ liệu đã được lưu vào '{PERSIST_PATH}'")
    print(f"📊 Tổng số chunks: {len(final_chunks)}")
    print(f"📚 Collection name: {COLLECTION_NAME}")
    print("=" * 60)

if __name__ == "__main__":
    main()