import os
import unicodedata
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["ANONYMIZED_TELEMETRY"] = "False"

class RAGService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self):
        print("🔥 Đang khởi tạo RAG Service (Optimized for GTX 1650)...")
        
        # 1. LLM (Giảm max_tokens để phản hồi nhanh hơn)
        LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llama:8080/v1")
        self.llm = ChatOpenAI(
            model="qwen1_5-1_8b-chat-q8_0",
            base_url=LLM_BASE_URL,
            api_key="not-needed",
            temperature=0.3,
            max_tokens=512, # Giảm xuống mức vừa đủ đọc
            model_kwargs={"stop": ["Question:", "Câu hỏi:", "<|im_end|>"]}
        )

        # 2. Embedding
        EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device}
        )

        # 3. Vectorstore
        # Đảm bảo dùng đường dẫn tuyệt đối để tránh lỗi path
        ABS_DATA_PATH = os.path.abspath("./core/data")
        PERSIST_PATH = "./core/chroma_db"
        
        self.vectorstore = Chroma(
            collection_name="law_docs",
            persist_directory=PERSIST_PATH,
            embedding_function=self.embedding_model
        )
        
        # Mapping từ khóa sang tên file (Lưu ý: Bạn cần đảm bảo tên file chính xác)
        self.topic_mapping = {
            "hôn nhân": ["Luật-Hôn-nhân-và-gia-đình.docx", "Nghị-quyết-326.docx"],
            "ly hôn": ["Luật-Hôn-nhân-và-gia-đình.docx", "Nghị-quyết-326.docx", "BLDS.docx"],
            "đất đai": ["Luật-Đất-đai.docx"],
            "hình sự": ["BLHS.docx", "BLTTHS.docx"],
            "tù": ["BLHS.docx", "BLTTHS.docx"],
            "dân sự": ["BLDS.docx"],
            "giao thông": ["LGTDB.docx", "ND168.docx"],
            "phạt nguội": ["ND168.docx"],
        }
        
        # Prompt ngắn gọn hơn để xử lý nhanh hơn
        self.prompt_template = ChatPromptTemplate.from_template("""Dựa vào luật sau:
---
{context}
---
Trả lời câu hỏi: {input}
(Chỉ trả lời dựa vào nội dung trên. Ngắn gọn, súc tích).""")

    def get_smart_filter(self, query):
        """Trả về list file tiềm năng dựa trên từ khóa"""
        query_lower = query.lower()
        target_files = set()
        
        # Lấy đường dẫn gốc để tạo filter path chính xác
        abs_data_path = os.path.abspath("./core/data")

        for keyword, filenames in self.topic_mapping.items():
            if keyword in query_lower:
                for fname in filenames:
                    # Tạo đường dẫn đầy đủ khớp với lúc Ingest
                    full_path = os.path.join(abs_data_path, fname)
                    target_files.add(full_path)
        
        if not target_files:
            return None
            
        if len(target_files) == 1:
            return {"source": {"$eq": list(target_files)[0]}}
        return {"source": {"$in": list(target_files)}}

    def query(self, question: str, k: int = 3):
        try:
            query = unicodedata.normalize("NFC", question.strip())
            print(f"📝 Query: {query}")

            # --- CHIẾN THUẬT SMART FILTER ---
            docs = []
            
            # Bước 1: Thử tìm với Filter (Nhanh nhất)
            metadata_filter = self.get_smart_filter(query)
            if metadata_filter:
                print("🎯 Đang tìm kiếm với Smart Filter...")
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": k, "filter": metadata_filter}
                )
                docs = retriever.invoke(query)
            
            # Bước 2: Fallback - Nếu không thấy docs nào, tìm toàn bộ (An toàn)
            if not docs:
                print("🌐 Filter không ra kết quả -> Tìm kiếm toàn bộ DB...")
                retriever_full = self.vectorstore.as_retriever(search_kwargs={"k": k})
                docs = retriever_full.invoke(query)

            print(f"✅ Tìm thấy {len(docs)} documents")
            
            if not docs:
                return {'answer': 'Không tìm thấy thông tin luật phù hợp.', 'sources': []}

            # Tạo context
            context = "\n\n".join([d.page_content for d in docs])
            
            # Gọi LLM
            formatted = self.prompt_template.format(context=context, input=query)
            messages = [
                SystemMessage(content="Bạn là trợ lý pháp luật."),
                HumanMessage(content=formatted)
            ]
            
            resp = self.llm.invoke(messages)
            answer = resp.content if hasattr(resp, "content") else str(resp)

            # Sources
            sources = [{'content': d.page_content[:150] + '...', 'metadata': d.metadata} for d in docs]

            return {'answer': answer, 'sources': sources}

        except Exception as e:
            print(f"❌ Error: {e}")
            return {'answer': 'Lỗi hệ thống.', 'sources': [], 'error': str(e)}

_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service