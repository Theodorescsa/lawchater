# Hướng dẫn chạy LawChat Django API với Docker

## 1. Cấu trúc thư mục

```
lawchater_be/
├── app_home/
│   ├── views.py          # ✅ API endpoints
│   └── ...
├── core/
│   ├── rag_service.py    # ✅ RAG Service (tạo mới)
│   ├── ingest.py         # ✅ Script nạp dữ liệu
│   ├── data/             # ✅ Các file luật .docx
│   ├── chroma_db/        # ✅ Vector database
│   ├── hf_cache/         # Cache HuggingFace
│   └── models/           # Model GGUF
├── lawchater_be/
│   ├── settings.py       # ✅ Cập nhật cấu hình
│   └── urls.py           # ✅ API routes
├── Dockerfile            # ✅ Docker config
├── docker-compose.yml    # ✅ Docker Compose
├── requirements.txt      # ✅ Dependencies
└── manage.py
```

## 2. Cập nhật code

### Bước 1: Tạo file mới `core/rag_service.py`
Copy code từ artifact `core/rag_service.py`

### Bước 2: Cập nhật `app_home/views.py`
Thay thế toàn bộ nội dung bằng code từ artifact `app_home/views.py`

### Bước 3: Cập nhật `lawchater_be/urls.py`
Thay thế toàn bộ nội dung bằng code từ artifact `lawchater_be/urls.py`

### Bước 4: Cập nhật `lawchater_be/settings.py`
Thêm vào cuối file các cấu hình từ artifact `lawchater_be/settings.py`

### Bước 5: Cập nhật các file Docker
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.dockerignore`

## 3. Chạy ứng dụng

### Bước 1: Build image
```bash
docker compose build
```

### Bước 2: Ingest dữ liệu (chỉ chạy 1 lần đầu)
```bash
docker compose run --rm web python core/ingest.py
```

### Bước 3: Khởi động services
```bash
docker compose up
```

Hoặc chạy background:
```bash
docker compose up -d
```

### Bước 4: Kiểm tra health
```bash
curl http://localhost:8000/api/health/
```

Response:
```json
{
  "status": "ok",
  "service": "LawChat API",
  "llm": "connected",
  "vectorstore": "ready"
}
```

## 4. Sử dụng API

### Endpoint: POST /api/chat/

**Request:**
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tôi ly hôn thì sẽ mất bao nhiêu tiền?",
    "k": 3
  }'
```

**Response:**
```json
{
  "question": "Tôi ly hôn thì sẽ mất bao nhiêu tiền?",
  "answer": "Theo Bộ luật Dân sự, chi phí ly hôn...",
  "sources": [
    {
      "content": "Điều 52. Chi phí tố tụng...",
      "metadata": {
        "source": "core/data/BLDS.docx"
      }
    }
  ]
}
```

### Test với Python:
```python
import requests

response = requests.post(
    'http://localhost:8000/api/chat/',
    json={
        'question': 'Điều 32 Bộ luật hình sự quy định gì?',
        'k': 3
    }
)

print(response.json())
```

### Test với JavaScript (Frontend):
```javascript
fetch('http://localhost:8000/api/chat/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'Tôi có thể khởi kiện ly hôn không?',
    k: 3
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## 5. Các lệnh hữu ích

### Xem logs
```bash
# Tất cả services
docker compose logs -f

# Chỉ Django
docker compose logs -f web

# Chỉ LLM
docker compose logs -f llama
```

### Restart services
```bash
docker compose restart
```

### Stop services
```bash
docker compose down
```

### Rebuild và restart
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### Chạy Django commands
```bash
# Migrate database
docker compose exec web python manage.py migrate

# Create superuser
docker compose exec web python manage.py createsuperuser

# Shell
docker compose exec web python manage.py shell
```

### Re-ingest dữ liệu
```bash
# Cách 1: Chạy script trực tiếp
docker compose exec web python core/ingest.py

# Cách 2: Gọi API (cần thêm authentication)
curl -X POST http://localhost:8000/api/ingest/
```

## 6. Kết nối Frontend

### CORS đã được cấu hình
Frontend có thể gọi API từ bất kỳ origin nào (development mode).

### Ví dụ React:
```javascript
// src/services/api.js
const API_URL = 'http://localhost:8000/api';

export const askQuestion = async (question) => {
  const response = await fetch(`${API_URL}/chat/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, k: 3 }),
  });
  
  return response.json();
};
```

## 7. Troubleshooting

### Lỗi "Module not found"
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### Lỗi GPU
Kiểm tra GPU có sẵn:
```bash
docker compose exec web python -c "import torch; print(torch.cuda.is_available())"
```

### Lỗi port đã được sử dụng
Thay đổi port trong `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Đổi từ 8000 sang 8001
```

### Reset toàn bộ
```bash
docker compose down -v
rm -rf core/chroma_db
docker compose build --no-cache
docker compose run --rm web python core/ingest.py
docker compose up
```

## 8. Production Deployment

### Sử dụng Gunicorn
Sửa `docker-compose.yml`:
```yaml
command: gunicorn lawchater_be.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

Thêm vào `requirements.txt`:
```
gunicorn>=21.2.0
```

### Tắt DEBUG mode
Trong `settings.py`:
```python
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'localhost']
```

### Cấu hình CORS cho production
```python
CORS_ALLOWED_ORIGINS = [
    "https://your-frontend.com",
]
```