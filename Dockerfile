# Sử dụng Python base image
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho việc compile C++ (cho llama.cpp)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# --- CÀI ĐẶT LLAMA-CPP-PYTHON VỚI CUDA SUPPORT ---
# Dòng này cực kỳ quan trọng để kích hoạt GPU
ENV CMAKE_ARGS="-DGGML_CUDA=on" 
ENV FORCE_CMAKE=1

# Cài đặt requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Mở port
EXPOSE 8000

# Command chạy mặc định (đã set trong docker-compose, có thể để trống hoặc set default)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]