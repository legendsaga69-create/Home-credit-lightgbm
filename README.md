# 🏦 Home Credit Default Risk — Hệ thống Chấm điểm Tín dụng End-to-End

> **ML Portfolio Project** | LightGBM · FastAPI · Streamlit · Docker  
> Mô hình dự báo nguy cơ vỡ nợ khách hàng, triển khai thành hệ thống API + giao diện sẵn sàng chạy local hoặc container.

---

## 📌 Tổng quan bài toán

Bộ dữ liệu **Home Credit Default Risk** (Kaggle) với mục tiêu dự báo nhị phân: khách hàng có khả năng **vỡ nợ (TARGET=1)** hay không trong quá trình sử dụng khoản vay.

| Thông tin | Chi tiết |
|---|---|
| Loại bài toán | Binary Classification |
| Dữ liệu | ~307,000 hồ sơ vay, 7 bảng dữ liệu liên quan |
| Tỉ lệ vỡ nợ (imbalance) | ~8.07% |
| Metric chính | AUC-ROC |

---

## 📊 Kết quả mô hình

| Metric | Giá trị |
|---|---|
| OOF AUC | **0.7881** |
| Gini Coefficient | **0.5762** |
| KS Statistic | **0.4329** |
| Phương pháp đánh giá | 5-Fold Stratified Cross-Validation |
| Số lượng features | 133 (sau SHAP filtering) |

---

## 🔁 Pipeline ML

### Bước 1 — EDA (Phân tích dữ liệu khám phá)
- Phân tích phân phối, missing values, và mối tương quan với TARGET trên toàn bộ 7 bảng dữ liệu
- Phân loại features theo nhóm: định tính, định lượng liên tục, nhị phân

### Bước 2 — Feature Engineering (Xây dựng đặc trưng)
Tổng hợp đặc trưng từ 6 bảng phụ (bureau, bureau_balance, previous_application, credit_card_balance, POS_CASH_balance, installments_payments) thành 8 nhóm:

| Nhóm | Mô tả |
|---|---|
| EXT_SOURCE | Aggregation và tương tác giữa 3 nguồn điểm tín dụng ngoài |
| Financial Health | Tỉ lệ nợ/thu nhập, LTV, annuity-to-income |
| DPD Cross-source | Chỉ số quá hạn tổng hợp từ nhiều nguồn |
| Stability Ratios | Độ ổn định việc làm, địa chỉ, điện thoại |
| Red Flag Indicators | Các chỉ báo rủi ro cao như tỉ lệ từ chối, lịch sử DPD |
| Bureau Aggregations | Thống kê từ lịch sử tín dụng tại Credit Bureau |
| Previous Application | Thống kê từ các hồ sơ vay trước đó |
| Installment Behavior | Hành vi trả góp: đúng hạn, trễ hạn, tỉ lệ hoàn thành |

### Bước 3 — Feature Selection (Lọc đặc trưng)
- Huấn luyện LightGBM baseline với ~265 features
- Tính **Median SHAP Value** trên toàn bộ tập OOF
- Loại bỏ features có Median SHAP ≤ 0 → còn lại **133 features**

### Bước 4 — Modeling (Huấn luyện mô hình)
- Thuật toán: **LightGBM Classifier**
- Tuning siêu tham số: **Optuna** (TPE Sampler)
- Đánh giá: **5-Fold Stratified Cross-Validation**
- Lưu artifact: 5 file `.pkl` (1 fold/file) + `model_metadata.json`

### Bước 5 — Deployment (Triển khai)
- **FastAPI**: Endpoint `/predict` (single) và `/predict_batch` (CSV upload)
- **Streamlit**: Dashboard chấm điểm và so sánh kết quả với nhãn thực tế
- **Docker**: Containerize toàn bộ hệ thống

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                    NGƯỜI DÙNG (Browser)                  │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP :8501
┌───────────────────────────▼─────────────────────────────┐
│              TẦNG FRONTEND (Streamlit :8501)             │
│  • Chọn hồ sơ mock → điều chỉnh EXT_SOURCE → chấm điểm │
│  • Upload CSV → batch prediction → download kết quả     │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP :8000
┌───────────────────────────▼─────────────────────────────┐
│              TẦNG BACKEND (FastAPI :8000)                │
│  POST /predict        — 1 hồ sơ JSON                    │
│  POST /predict_batch  — CSV upload, N records            │
│  GET  /               — Health check                     │
└───────────────────────────┬─────────────────────────────┘
                            │ joblib.load
┌───────────────────────────▼─────────────────────────────┐
│              TẦNG MODEL (Artifacts)                      │
│  lgb_fold_1.pkl ~ lgb_fold_5.pkl                        │
│  model_metadata.json (133 feature names + cat_cols)     │
└─────────────────────────────────────────────────────────┘
```

**Logic Ensemble Inference:**
```
Input JSON/CSV
    → Ép kiểu đúng thứ tự 133 features
    → 5 mô hình × predict_proba[:, 1]
    → Trung bình cộng xác suất
    → So sánh với threshold → Approve / Reject
```

---

## 📁 Cấu trúc thư mục

```
E2E - Home Credit/
├── backend/
│   ├── main.py              # FastAPI app: load model, inference logic, API endpoints
│   ├── artifacts/           # Model artifacts (copy vào đây trước khi chạy)
│   │   ├── lgb_fold_1.pkl ~ lgb_fold_5.pkl
│   │   ├── model_metadata.json
│   │   └── mock_samples.json
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit dashboard
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🚀 Hướng dẫn chạy

### Yêu cầu
- Python 3.10+
- Docker & Docker Compose (nếu chạy bằng container)

### Cách 1 — Chạy local (không Docker)

**Bước 1:** Cài đặt thư viện

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend (terminal mới)
cd frontend
pip install -r requirements.txt
```

**Bước 2:** Copy artifacts vào đúng vị trí

```
backend/artifacts/lgb_fold_1.pkl
backend/artifacts/lgb_fold_2.pkl
backend/artifacts/lgb_fold_3.pkl
backend/artifacts/lgb_fold_4.pkl
backend/artifacts/lgb_fold_5.pkl
backend/artifacts/model_metadata.json
backend/artifacts/mock_samples.json
```

**Bước 3:** Khởi động Backend (Terminal 1)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Bước 4:** Khởi động Frontend (Terminal 2)

```bash
cd frontend
streamlit run app.py
```

Truy cập giao diện tại: [http://localhost:8501](http://localhost:8501)  
API documentation tại: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Cách 2 — Chạy bằng Docker Compose

```bash
# Build và khởi động toàn bộ hệ thống
docker-compose up --build

# Chạy nền
docker-compose up --build -d

# Dừng hệ thống
docker-compose down
```

Truy cập giao diện tại: [http://localhost:8501](http://localhost:8501)

---

## 🖥️ Hướng dẫn sử dụng giao diện

### Chấm điểm đơn lẻ (Mock Data)
1. Chọn hồ sơ khách hàng từ dropdown
2. Điều chỉnh thử nghiệm các biến EXT_SOURCE_1/2/3
3. Bấm **"Chấm điểm tín dụng"**
4. Xem kết quả: xác suất vỡ nợ, quyết định Duyệt/Từ chối, so sánh với TARGET thực tế

### Chấm điểm hàng loạt (Batch Prediction)
1. Upload file CSV (định dạng giống `test_processed.csv`, có 133 features)
2. Bấm **"Chấm điểm toàn bộ"**
3. Xem bảng kết quả và tải về file CSV

---

## 🔧 API Reference

### `GET /`
Health check

```json
{"status": "healthy", "message": "API Chấm điểm tín dụng đang hoạt động."}
```

### `POST /predict`
Chấm điểm 1 hồ sơ

**Request body:** JSON với 133 features (xem `model_metadata.json`)

**Response:**
```json
{
  "credit_risk_probability": 0.3241,
  "decision": "Approve",
  "threshold_rule": 0.5,
  "fold_details": [0.31, 0.33, 0.32, 0.34, 0.31]
}
```

### `POST /predict_batch`
Chấm điểm hàng loạt từ file CSV

**Request:** `multipart/form-data` với field `file` là file CSV

**Response:**
```json
{
  "total_records": 48744,
  "results": [
    {"record_index": 0, "credit_risk_probability": 0.1823, "decision": "Approve"},
    {"record_index": 1, "credit_risk_probability": 0.6741, "decision": "Reject"}
  ]
}
```

> **Lưu ý khi chạy Docker:** `app.py` đọc URL từ biến môi trường `BACKEND_URL` và `BACKEND_BATCH_URL`. Khi chạy bằng Docker Compose, các biến này đã được inject tự động — Streamlit sẽ gọi backend qua DNS nội bộ `http://backend:8000` thay vì `localhost`.

---

## 🛠️ Stack công nghệ

| Layer | Công nghệ |
|---|---|
| ML Framework | LightGBM |
| Hyperparameter Tuning | Optuna |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Serialization | Joblib |
| Containerization | Docker + Docker Compose |
| Data Processing | Pandas, NumPy |
| Model Validation | Pydantic |

---

## 📝 Ghi chú kỹ thuật

- **Missing values:** Không impute — LightGBM xử lý NaN natively thông qua cơ chế split tìm kiếm hướng tốt nhất cho giá trị thiếu
- **Categorical features:** Ép về dtype `category` trước khi inference để đảm bảo nhất quán với lúc training
- **Ensemble:** Trung bình cộng xác suất từ 5 fold giúp giảm variance so với single model
- **Threshold mặc định:** 0.5 — có thể điều chỉnh tùy theo yêu cầu nghiệp vụ (ví dụ: hạ về ~0.08 để match tỉ lệ vỡ nợ thực tế của dataset)
