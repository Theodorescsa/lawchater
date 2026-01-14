# core/app.py

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

    # --- CẤU HÌNH DỮ LIỆU TĨNH (MAPPING & HINTS) ---
    # [QUAN TRỌNG] Tên file ở đây phải khớp với kết quả check_sources.py (KHÔNG CÓ .docx)
    PRIMARY_MAPPING = {
        "đánh người": ("134", "BLHS"), "thương tích": ("134", "BLHS"), "nhập viện": ("134", "BLHS"),
        "tự vệ": ("22", "BLHS"), "đánh trả": ("22", "BLHS"), "phòng vệ": ("22", "BLHS"),
        "không báo công an": ("19", "BLHS"), "im lặng": ("19", "BLHS"), "nể tình": ("19", "BLHS"),
        "che giấu": ("18", "BLHS"),
        "say rượu": ("13", "BLHS"), "bia rượu": ("13", "BLHS"), "xỉn": ("13", "BLHS"),
        "trẻ em": ("12", "BLHS"), "14 tuổi": ("12", "BLHS"), "15 tuổi": ("12", "BLHS"),
        "mượn xe": ("17", "BLHS"), "vạ lây": ("17", "BLHS"), "đồng phạm": ("17", "BLHS"),
        "mua đồ ăn trộm": ("323", "BLHS"), "tiêu thụ": ("323", "BLHS"), "đồ gian": ("323", "BLHS"),
        "lãi suất cao": ("201", "BLHS"), "nặng lãi": ("201", "BLHS"), "lãi cắt cổ": ("201", "BLHS"),
        "đánh bài": ("321", "BLHS"), "đánh bạc": ("321", "BLHS"), "tá lả": ("321", "BLHS")
    }

    # Gợi ý cứng từ chuyên gia để định hướng LLM
    HINTS = {
        "19": "Theo Điều 19 Bộ luật Hình sự (BLHS): Hành vi không tố giác tội phạm (im lặng) là VI PHẠM PHÁP LUẬT. Hãy khẳng định người hỏi CÓ TỘI.",
        "13": "Theo Điều 13 Bộ luật Hình sự (BLHS): Phạm tội trong tình trạng say rượu VẪN PHẢI CHỊU TRÁCH NHIỆM HÌNH SỰ. Say rượu không phải là tình tiết giảm nhẹ.",
        "12": "Theo Điều 12 Bộ luật Hình sự (BLHS): Người từ đủ 14 đến dưới 16 tuổi phải chịu trách nhiệm về tội RẤT NGHIÊM TRỌNG. Nếu chỉ đánh nhau nhẹ hoặc gây rối thì thường xử lý hành chính.",
        "22": "Theo Điều 22 Bộ luật Hình sự (BLHS): Đánh trả khi đang bị tấn công là Phòng vệ chính đáng. Nhưng đánh khi kẻ trộm đã bỏ chạy là Vượt quá giới hạn.",
        "321": "Theo Điều 321 Bộ luật Hình sự (BLHS): Đánh bạc trên 5 triệu đồng mới bị xử lý hình sự. Dưới mức này phạt hành chính.",
        "17": "Theo Điều 17 Bộ luật Hình sự (BLHS): Cho mượn xe mà KHÔNG BIẾT bạn đi gây án thì KHÔNG PHẢI ĐỒNG PHẠM.",
        "201": "Theo Điều 201 Bộ luật Hình sự (BLHS): Lãi suất trên 100%/năm VÀ thu lợi > 30 triệu mới bị xử lý Hình sự.",
        "323": "Theo Điều 323 Bộ luật Hình sự (BLHS): Chỉ phạm tội tiêu thụ đồ gian nếu BIẾT RÕ đó là tài sản phạm tội. Nếu vô ý không biết thì không phạm tội hình sự."
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self):
        print(" Đang khởi tạo LawChatter RAG Engine (LlamaCpp + Rerank + Pinning)...")
        
        self.BASE_DIR = Path(__file__).resolve().parent
        self.MODEL_FILENAME = "qwen2.5-3b-instruct-q5_k_m.gguf"
        self.MODEL_PATH = str(self.BASE_DIR / "models" / self.MODEL_FILENAME)
        
        self.PERSIST_PATH = str(self.BASE_DIR / "chroma_db")
        self.EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
        self.RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.COLLECTION_NAME = "law_docs"

        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f" Lỗi: Không tìm thấy model tại {self.MODEL_PATH}")

        # 1. Khởi tạo LLM
        print(f" Loading LLM from {self.MODEL_PATH}...")
        self.llm = LlamaCpp(
            model_path=self.MODEL_PATH,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=8192,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            verbose=False,
            stop=["<|im_end|>", "User:", "Kết thúc"]
        )

        # 2. Embedding & Vector Store
        print(" Loading Embedding Model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL_NAME, 
            model_kwargs={"device": "cpu"}
        )
        
        self.vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.PERSIST_PATH,
            embedding_function=self.embedding_model
        )

        # 3. Reranker
        print(" Loading Reranker Model...")
        self.reranker = CrossEncoder(self.RERANK_MODEL_NAME)

        # 4. Prompt Templates
        self.rewrite_template = """<|im_start|>system
Bạn là chuyên gia pháp lý. Nhiệm vụ: Trả về danh sách TỪ KHÓA + CẶP ĐỊNH DANH [Điều luật | Tên file].
VÍ DỤ:
User: Say rượu đánh người
Output: [Điều 134 | BLHS], [Điều 13 | BLHS]
User: {question}
Output:<|im_end|>
<|im_start|>assistant
"""
        self.rewrite_prompt = PromptTemplate(input_variables=["question"], template=self.rewrite_template)

        self.answer_template = """<|im_start|>system
Bạn là luật sư AI. Dựa vào [GỢI Ý CỦA CHUYÊN GIA] và Context để trả lời.

QUY TẮC QUAN TRỌNG:
1. Đọc [GỢI Ý] và [CONTEXT].
1. Nếu có GỢI Ý, dùng nó làm kết luận chính.
2. NHÌN KỸ "Nguồn" trong Context.
3. Trích dẫn số Điều luật chính xác.
2. Trả lời theo đúng cấu trúc bên dưới.
3. TUYỆT ĐỐI KHÔNG lặp lại nội dung đã viết.

<|im_end|>
<|im_start|>user
CONTEXT:
{context}

[GỢI Ý CỦA CHUYÊN GIA]:
{hints}

CÂU HỎI: {question}
<|im_end|>
<|im_start|>assistant
"""
        self.qa_prompt = PromptTemplate(input_variables=["context", "hints", "question"], template=self.answer_template)

        print(" RAG Service Ready!")

    # --- HELPER FUNCTIONS ---
    def clean_text(self, text):
        text = text.replace("passage: ", "")
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[-_=*]{3,}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_source_filter(self, query):
        """Logic lọc file thông minh - ĐÃ CẬP NHẬT TÊN FILE KHÔNG ĐUÔI"""
        query_lower = query.lower()
        target_files = set()

        # --- CẬP NHẬT TÊN FILE CHÍNH XÁC (KHÔNG CÓ .docx/.pdf) ---
        f_blds = "BLDS"
        f_blhs = "BLHS"
        f_bltths = "BLTTHS"
        f_lgtdb = "LGTDB"
        f_anm = "Luật an ninh mạng"
        f_nd168 = "ND168"
        f_nd53 = "Nghị định 53 - ND - CP"

        keyword_map = {
            "an_ninh_mang": (
                [f_anm, f_nd53], 
                ["an ninh mạng", "hacker", "virus", "mã độc", "dữ liệu cá nhân", "xúc phạm", "facebook", "zalo", "nghị định 53"]
            ),
            "giao_thong": (
                [f_lgtdb, f_nd168], 
                ["giao thông", "lgtđb", "lái xe", "đèn đỏ", "nồng độ cồn", "rượu bia", "tước bằng", "phạt nguội"]
            ),
            "hinh_su": (
                [f_blhs, f_bltths], 
                ["hình sự", "blhs", "tù", "giết người", "trộm cắp", "cướp", "lừa đảo", "ma túy", "đánh bạc", "khởi tố", "bị can", "thương tích", "trẻ em", "vị thành niên"]
            ),
            "dan_su": (
                [f_blds], 
                ["dân sự", "blds", "hợp đồng", "bồi thường", "thừa kế", "di chúc", "đất đai", "ly hôn", "vay nợ"]
            )
        }

        for _, (files, keywords) in keyword_map.items():
            if any(k in query_lower for k in keywords):
                target_files.update(files)
        
        if not target_files:
            return None
        
        target_list = list(target_files)
        if len(target_list) == 1:
            return {"source_name": {"$eq": target_list[0]}}
        return {"source_name": {"$in": target_list}}

    def rewrite_query(self, user_query):
        try:
            prompt = self.rewrite_prompt.format(question=user_query)
            result = self.llm.invoke(prompt)
            legal_keywords = re.sub(r"^(Output|Trả lời|Keywords):\s*", "", result.strip(), flags=re.IGNORECASE)
            legal_keywords = legal_keywords.replace("\n", " ")
            return legal_keywords
        except Exception:
            return user_query

    def advanced_retrieval(self, search_query, rank_query, primary_targets, metadata_filter, top_k=5):
        # 1. SEMANTIC SEARCH
        semantic_docs = self.vectorstore.similarity_search(
            f"query: {search_query}", 
            k=30, 
            filter=metadata_filter
        )

        # 2. FORCE INJECTION (Ghim tài liệu)
        pinned_docs = []
        if primary_targets:
            print(f" [Force Injection] Targets: {primary_targets}")
            for art_num, target_file in primary_targets:
                queries = [f"Điều {art_num}", f"nội dung điều {art_num}"]
                specific_filter = {"source_name": target_file} if target_file else metadata_filter
                
                for q in queries:
                    found = self.vectorstore.similarity_search(f"query: {q}", k=5, filter=specific_filter)
                    for doc in found:
                        match_pattern = rf"điều\s*[._-]*\s*{art_num}(?:\D|$)"
                        if re.search(match_pattern, doc.page_content, re.IGNORECASE):
                            doc.metadata['is_pinned'] = True
                            pinned_docs.append(doc)

        # 3. MERGE & CLEAN
        unique_docs = {}
        all_raw_docs = pinned_docs + semantic_docs 
        
        cleaned_doc_objects = []
        for doc in all_raw_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
                doc.page_content = self.clean_text(doc.page_content)
                cleaned_doc_objects.append(doc)

        if not cleaned_doc_objects:
            return []

        # 4. RANKING
        final_pinned = []
        normal_docs = []
        for doc in cleaned_doc_objects:
            if doc.metadata.get('is_pinned', False):
                final_pinned.append(doc)
            else:
                normal_docs.append(doc)
        
        if normal_docs:
            doc_contents = [d.page_content for d in normal_docs]
            pairs = [[rank_query, content] for content in doc_contents]
            scores = self.reranker.predict(pairs)
            
            scored_normal = []
            for doc, score in zip(normal_docs, scores):
                doc.metadata['score'] = float(score)
                scored_normal.append((doc, float(score)))
            
            scored_normal.sort(key=lambda x: x[1], reverse=True)
            normal_docs = [d for d, s in scored_normal if s > -10.0]

        final_docs = final_pinned + normal_docs[:top_k]
        return final_docs

    def query(self, question: str, k: int = 3):
        try:
            query_str = unicodedata.normalize("NFC", question.strip())
            
            primary_targets = set()
            active_hints = []
            
            for kw, target in self.PRIMARY_MAPPING.items():
                if kw in query_str.lower():
                    primary_targets.add(target)
                    art_num = target[0]
                    if art_num in self.HINTS:
                        active_hints.append(f"- Về Điều {art_num}: {self.HINTS[art_num]}")
            
            hint_text = "\n".join(active_hints) if active_hints else "Không có gợi ý đặc biệt."
            
            legal_keywords = self.rewrite_query(query_str)
            
            full_scan_text = f"{query_str} . {legal_keywords}"
            metadata_filter = self.get_source_filter(full_scan_text)
            
            docs = self.advanced_retrieval(legal_keywords, query_str, primary_targets, metadata_filter, top_k=k)
            
            if not docs:
                return {'answer': 'Không tìm thấy thông tin luật phù hợp trong cơ sở dữ liệu.', 'sources': []}

            context_text = ""
            for doc in docs:
                source = doc.metadata.get('source_name', 'Unknown')
                tag = "[GỢI Ý QUAN TRỌNG]" if doc.metadata.get('is_pinned') else ""
                context_text += f"--- NGUỒN: {source} {tag} ---\n{doc.page_content}\n\n"

            formatted_prompt = self.qa_prompt.format(context=context_text, hints=hint_text, question=query_str)
            answer = self.llm.invoke(formatted_prompt)

            sources = []
            for d in docs:
                sources.append({
                    'content': self.clean_text(d.page_content)[:200] + '...',
                    'metadata': d.metadata,
                    'is_pinned': d.metadata.get('is_pinned', False)
                })

            return {'answer': answer, 'sources': sources}

        except Exception as e:
            print(f" Error RAG: {e}")
            import traceback
            traceback.print_exc()
            return {'answer': 'Lỗi hệ thống khi xử lý câu hỏi.', 'sources': [], 'error': str(e)}

_rag_service = None

def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service