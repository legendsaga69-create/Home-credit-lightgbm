# Sử dụng nền tảng Python tương đồng với Backend để tránh xung đột thư viện đám mây
FROM python:3.11-slim

WORKDIR /app

# Sao chép và tiến hành cấu hình cài đặt thư viện cho giao diện Web
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Đổ toàn bộ mã nguồn tầng hiển thị vào môi trường làm việc
COPY . .

# Mở cổng kết nối 8501 mặc định của nền tảng Streamlit
EXPOSE 8501

# Lệnh khởi chạy giao diện Web, cấu hình tường minh địa chỉ IP để Docker Compose phân quyền mạng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]