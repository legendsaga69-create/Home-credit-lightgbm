# Sử dụng phiên bản Python slim để giảm dung lượng file build hệ thống
FROM python:3.11-slim

# Thiết lập thư mục làm việc mặc định bên trong môi trường ảo của Container
WORKDIR /app

# BẮT BUỘC: Cài đặt thư viện xử lý đa luồng OpenMP (libgomp1) cho thuật toán LightGBM
# Nếu thiếu thư viện này, tầng biên dịch C++ của LightGBM trên Linux sẽ bị lỗi sập hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file định danh thư viện vào trước để tận dụng cơ chế lưu cache của Docker
COPY requirements.txt .

# Tiến hành cài đặt các thư viện Python, tắt chế độ lưu cache cài đặt để tối ưu bộ nhớ
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn và các file mô hình artifacts vào trong Container
COPY . .

# Khai báo cổng kết nối mạng mạng mà container sẽ mở ra bên ngoài
EXPOSE 8000

# Lệnh kích hoạt FastAPI Server chạy trên toàn bộ giao diện mạng của container tại cổng 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]