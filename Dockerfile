# 1. Image CUDA Development Kit (Chuẩn cho LlamaCpp)
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# --- QUAN TRỌNG: Cấu hình để không bị hỏi Timezone khi cài đặt ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# 2. Cài đặt Python và TZDATA (Sửa lỗi thiếu múi giờ)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    pkg-config \
    default-libmysqlclient-dev \
    tzdata \ 
    && rm -rf /var/lib/apt/lists/*

# Cấu hình lại múi giờ hệ thống (đảm bảo container chạy đúng giờ VN)
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Tạo symlink: gõ 'python' là hiểu là python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .

# 3. Cấu hình biến môi trường cho CUDA
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# --- FIX LỖI LINKING LIBCUDA CHO LLAMA-CPP ---
# Thêm đường dẫn stubs
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs

# Hack symlink: Trình biên dịch tìm .so.1 nhưng stubs chỉ có .so
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Cài đặt llama-cpp-python trước (Tận dụng cache docker layer cho việc build lâu này)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir llama-cpp-python

# Cài các gói còn lại
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]