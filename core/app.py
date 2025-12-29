import os
import re
import unicodedata
from pathlib import Path

# AI Libraries
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

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
        print("🔥 Đang khởi tạo LawChatter RAG Engine (LlamaCpp + Rerank)...")
        
        # --- CẤU HÌNH ĐƯỜNG DẪN ---
        # Lấy đường dẫn thư mục 'core' hiện tại
        self.BASE_DIR = Path(__file__).resolve().parent
        
        # Cập nhật đường dẫn MODEL tại core/models/
        self.MODEL_FILENAME = "qwen2.5-3b-instruct-q5_k_m.gguf" # <-- Tên file model của bạn
        self.MODEL_PATH = str(self.BASE_DIR / "models" / self.MODEL_FILENAME)
        
        self.PERSIST_PATH = str(self.BASE_DIR / "chroma_db")
        self.EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
        self.RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.COLLECTION_NAME = "law_docs"

        # Kiểm tra file model tồn tại chưa
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"❌ Lỗi: Không tìm thấy model tại {self.MODEL_PATH}. Vui lòng kiểm tra folder core/models/")

        # 1. Khởi tạo LLM (LlamaCpp)
        print(f"⏳ Loading LLM from {self.MODEL_PATH}...")
        self.llm = LlamaCpp(
            model_path=self.MODEL_PATH,
            n_gpu_layers=-1,      # Đẩy 100% layers lên GPU
            n_batch=512,
            n_ctx=4096,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.15,
            verbose=False,        # Tắt log rác trong console Django
            stop=["<|im_end|>", "Người dùng:", "Kết thúc"]
        )

        # 2. Embedding & Vector Store
        print("⏳ Loading Embedding Model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL_NAME, 
            model_kwargs={"device": "cpu"} # ChromaDB chạy CPU để tiết kiệm VRAM cho LLM
        )
        
        self.vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.PERSIST_PATH,
            embedding_function=self.embedding_model
        )

        # 3. Reranker (Cross-Encoder)
        print("⏳ Loading Reranker Model...")
        self.reranker = CrossEncoder(self.RERANK_MODEL_NAME)

        # 4. Prompt Template (Đã tối ưu cho Qwen/Llama)
        template = """<|im_start|>system
Bạn là trợ lý pháp lý ảo LawChatter.

NHIỆM VỤ:
Sử dụng thông tin trong thẻ <documents> để trả lời câu hỏi.

QUY TRÌNH SUY LUẬN:
1. Xác định đúng đoạn văn bản chứa câu trả lời.
2. Trích xuất "Số hiệu điều luật" (VD: Điều 168, Điều 20...).
3. Tổng hợp đầy đủ quy định/khung hình phạt.

YÊU CẦU ĐẦU RA:
- Bắt đầu câu trả lời bằng: "Theo quy định tại [Số hiệu điều luật]..."
- Trình bày chi tiết, gạch đầu dòng rõ ràng.
- Kết thúc bằng: (Nguồn: [Tên_File])

VÍ DỤ MẪU:
Câu hỏi: Tội cướp tài sản bị phạt thế nào?
Trả lời:
Theo quy định tại Điều 168 Bộ luật Hình sự:
- Người nào dùng vũ lực đe dọa chiếm đoạt tài sản thì bị phạt tù từ 03 năm đến 10 năm.
(Nguồn: BLHS.docx)
<|im_end|>
<|im_start|>user
DỮ LIỆU LUẬT (XML):
{context}

Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
Câu trả lời:"""
        self.prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

        print("✅ RAG Service Ready!")

    # --- CÁC HÀM HỖ TRỢ (HELPER) ---
    def clean_text(self, text):
        text = text.replace("passage: ", "") # Loại bỏ prefix của E5
        text = re.sub(r'[-_=*]{3,}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_source_filter(self, query):
        """Logic lọc file thông minh dựa trên từ khóa"""
        query_lower = query.lower()
        target_files = set()

        # Định nghĩa mapping file
        f_blds = "BLDS.docx"
        f_blhs = "BLHS.docx"
        f_bltths = "BLTTHS.docx"
        f_lgtdb = "LGTDB.docx"
        f_anm = "Luật-An-Ninh-Mạng.docx"
        f_nd168 = "ND168.docx"
        f_nd53 = "Nghị-định-53-ND-CP.docx"

        keyword_map = {
            "an_ninh_mang": ([f_anm, f_nd53], ["an ninh mạng", "hacker", "virus", "mã độc", "dữ liệu cá nhân", "xúc phạm", "facebook", "zalo"]),
            "giao_thong": ([f_lgtdb, f_nd168], ["giao thông", "lgtđb", "lái xe", "đèn đỏ", "nồng độ cồn", "rượu bia", "tước bằng", "phạt nguội", "mũ bảo hiểm"]),
            "hinh_su": ([f_blhs, f_bltths], ["hình sự", "blhs", "tù", "giết người", "trộm cắp", "cướp", "lừa đảo", "ma túy", "đánh bạc", "khởi tố", "bị can"]),
            "dan_su": ([f_blds], ["dân sự", "blds", "hợp đồng", "bồi thường", "thừa kế", "di chúc", "đất đai", "ly hôn", "vay nợ"])
        }

        for _, (files, keywords) in keyword_map.items():
            if any(k in query_lower for k in keywords):
                target_files.update(files)
        
        if not target_files:
            return None
        
        target_list = list(target_files)
        # Cú pháp lọc của ChromaDB
        if len(target_list) == 1:
            return {"source_name": {"$eq": target_list[0]}}
        return {"source_name": {"$in": target_list}}

    def advanced_retrieval(self, query, metadata_filter, top_k_final=3):
        """Vector Search -> Cross-Encoder Rerank"""
        # B1: Lấy rộng (top 15)
        initial_docs = self.vectorstore.similarity_search(
            f"query: {query}", 
            k=15, 
            filter=metadata_filter
        )
        if not initial_docs: return []

        # B2: Rerank
        doc_contents = [self.clean_text(d.page_content) for d in initial_docs]
        pairs = [[query, content] for content in doc_contents]
        scores = self.reranker.predict(pairs)
        
        # B3: Sort & Filter
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        final_docs = []
        
        for doc, score in scored_docs[:top_k_final]:
            if score > -5.0: # Ngưỡng chấp nhận
                doc.metadata['score'] = float(score)
                final_docs.append(doc)
                
        return final_docs

    # --- HÀM CHÍNH ĐƯỢC API GỌI ---
    def query(self, question: str, k: int = 3):
        try:
            query_str = unicodedata.normalize("NFC", question.strip())
            
            # 1. Tìm tài liệu
            metadata_filter = self.get_source_filter(query_str)
            docs = self.advanced_retrieval(query_str, metadata_filter, top_k_final=k)
            
            if not docs:
                return {'answer': 'Không tìm thấy thông tin luật phù hợp trong cơ sở dữ liệu.', 'sources': []}

            # 2. Tạo Context XML
            context_text = "<documents>\n"
            for i, doc in enumerate(docs):
                clean_content = self.clean_text(doc.page_content)
                source = doc.metadata.get('source_name', 'Unknown')
                context_text += f'<doc id="{i+1}" source="{source}">\n{clean_content}\n</doc>\n'
            context_text += "</documents>"

            # 3. Gọi LLM trả lời
            formatted_prompt = self.prompt_template.format(context=context_text, question=query_str)
            answer = self.llm.invoke(formatted_prompt)

            # 4. Format nguồn để trả về API
            sources = []
            for d in docs:
                sources.append({
                    'content': self.clean_text(d.page_content)[:200] + '...',
                    'metadata': d.metadata
                })

            return {'answer': answer, 'sources': sources}

        except Exception as e:
            print(f"❌ Error RAG: {e}")
            import traceback
            traceback.print_exc()
            return {'answer': 'Lỗi hệ thống khi xử lý câu hỏi.', 'sources': [], 'error': str(e)}

_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service