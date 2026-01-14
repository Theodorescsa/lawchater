# 1. Image CUDA Development Kit
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# --- LẤY UV TỪ IMAGE CHÍNH HÃNG ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# --- CẤU HÌNH MÔI TRƯỜNG ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV UV_SYSTEM_PYTHON=1 

# 2. Cài đặt Python và các tool biên dịch
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    build-essential \
    cmake \
    git \
    pkg-config \
    default-libmysqlclient-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# --- FIX LỖI "python: not found" ---
RUN ln -s /usr/bin/python3 /usr/bin/python

# Cấu hình múi giờ
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /app

COPY requirements.txt .

# 3. Cấu hình biến môi trường cho CUDA (LlamaCPP)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# --- [QUAN TRỌNG: ĐÃ XÓA DÒNG ENV LD_LIBRARY_PATH Ở ĐÂY] ---
# Chỉ giữ lại symlink để hỗ trợ build (nếu cần)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# 4. CÀI ĐẶT BẰNG UV
# Bước 4a: Cài llama-cpp-python
# --- [SỬA LẠI: CHỈ TRỎ VÀO STUBS TRONG CÂU LỆNH NÀY THÔI] ---
# Lúc build xong thì biến môi trường này tự mất -> Lúc chạy sẽ dùng driver thật
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs uv pip install --no-cache llama-cpp-python

# Bước 4b: Cài các gói còn lại
RUN uv pip install --no-cache -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]