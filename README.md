# Home Credit Default Risk — Dual-Method Credit Scoring System

> **ML Portfolio Project** | LightGBM · WoE/IV · Logistic Regression · Monte Carlo · FastAPI · Docker  
> Triển khai và so sánh hai phương pháp chấm điểm tín dụng trên cùng 307K hồ sơ vay: ML pipeline hiện đại và Scorecard truyền thống chuẩn ngân hàng.

---

## Tổng quan bài toán

Bộ dữ liệu **Home Credit Default Risk** (Kaggle) với mục tiêu dự báo nhị phân: khách hàng có khả năng **vỡ nợ (TARGET=1)** hay không.

| Thông tin | Chi tiết |
|---|---|
| Loại bài toán | Binary Classification |
| Dữ liệu | ~307,000 hồ sơ vay, 7 bảng dữ liệu liên quan |
| Tỉ lệ vỡ nợ (class imbalance) | ~8.07% |
| Metric chính | AUC-ROC, Gini, KS |

---

## Kết quả tổng hợp

| Metric | LightGBM Ensemble | Scorecard (WoE + LR) |
|---|---|---|
| AUC | **0.7881** | 0.7118 |
| Gini | **0.5762** | 0.4237 |
| KS | **43.29%** | 31.32% (Bootstrap 95% CI: 30.04%–32.74%) |
| PSI (Train vs Test) | — | 0.0001 (ổn định) |
| Interpretability | ⚠️ SHAP post-hoc | ✅ Hệ số β tường minh |
| Regulatory Audit | ❌ Black box | ✅ Phù hợp Basel II |

---

## Pipeline 1 — LightGBM (ML Approach)

### Bước 1 — Feature Engineering

Tổng hợp đặc trưng từ 6 bảng phụ thành **8 nhóm feature**:

| Nhóm | Mô tả |
|---|---|
| EXT_SOURCE | Aggregation và tương tác 3 nguồn điểm tín dụng ngoài (mean, std, product) |
| Financial Health | LTV, debt-to-income, annuity-to-income |
| DPD Cross-source | Chỉ số quá hạn tổng hợp từ nhiều nguồn |
| Stability Ratios | Độ ổn định việc làm, địa chỉ, điện thoại |
| Red Flag Indicators | Tỉ lệ từ chối, lịch sử DPD |
| Bureau Aggregations | Thống kê lịch sử tín dụng: debt ratio, active rate |
| Previous Application | Approval rate, credit/goods ratio |
| Installment Behavior | Tỉ lệ trễ hạn, payment ratio, completion rate |

### Bước 2 — Feature Selection (Median SHAP)

- Train LightGBM baseline với ~265 features
- Tính **Median |SHAP Value|** trên toàn bộ OOF predictions
- Loại features có Median SHAP ≤ 0 → còn **133 features**

> SHAP ưu việt hơn feature importance mặc định vì đo đóng góp thực tế từng feature trong bối cảnh toàn model, bao gồm cả tương tác phi tuyến.

### Bước 3 — Modeling & Tuning

- **Thuật toán**: LightGBM Classifier (Leaf-wise tree growth)
- **Tuning**: Optuna TPE Sampler (Bayesian optimization)
- **Đánh giá**: 5-Fold Stratified Cross-Validation → OOF predictions
- **Ensemble**: Trung bình cộng xác suất từ 5 fold → giảm variance

**Kết quả:**

| Metric | Giá trị |
|---|---|
| OOF AUC | **0.7881** |
| Gini | **0.5762** |
| KS | **43.29%** |

---

## Pipeline 2 — Traditional Scorecard (WoE/IV + Logistic Regression)

### Bước 1 — Lọc biến sơ bộ

| Bước lọc | Trước | Sau |
|---|---|---|
| Missing rate > 50% | 229 cột | 160 cột |
| IV < 0.02 | 160 cột | 57 cột |
| Stepwise VIF > 5 | 57 cột | 52 cột |
| WoE variance = 0 | 52 cột | 48 cột |
| p-value ≥ 0.05 | 48 cột | **39 cột** |

### Bước 2 — WoE / IV Binning

**Weight of Evidence (WoE):**

$$WoE_i = \ln\left(\frac{\%Good_i}{\%Bad_i}\right)$$

**Information Value (IV):**

$$IV = \sum_{i=1}^{n} (\%Good_i - \%Bad_i) \times WoE_i$$

Kỹ thuật áp dụng:
- **Equal-frequency binning** (quantile): đảm bảo mỗi bin đủ observation
- **Laplace Smoothing**: tránh `log(0)` khi bin có `n_bad = 0`
- **Fit/Transform tách biệt**: fit chỉ trên train set → tránh data leakage

### Bước 3 — Stepwise VIF

Tính VIF trên `df_woe` (không phải giá trị gốc) để phản ánh đúng đa cộng tuyến đi vào model:

$$VIF_i = \frac{1}{1 - R^2_i}$$

Loại từng biến VIF cao nhất, tính lại đến khi tất cả VIF ≤ 5.

### Bước 4 — Logistic Regression

$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 WoE_1 + \beta_2 WoE_2 + \ldots$$

Fit bằng `statsmodels.Logit` (Newton-Raphson), loại biến p-value ≥ 0.05.

### Bước 5 — Score Scaling (300–850)

$$Score = Offset - Factor \times \ln(odds), \quad Factor = \frac{PDO}{\ln 2}$$

| Tham số | Giá trị |
|---|---|
| Base Score | 600 |
| Target Odds | 50:1 |
| PDO | 20 |
| Score range thực tế | 477–650 |

**Decile Analysis** — default rate đơn điệu hoàn toàn:

| Decile | Score | Default Rate |
|---|---|---|
| 1 (rủi ro nhất) | 477–535 | 21.79% |
| 5 | 560–565 | 6.86% |
| 10 (an toàn nhất) | 596–650 | 1.51% |

### Bước 6 — Kiểm định & Giám sát

| Kiểm định | Kết quả | Đánh giá |
|---|---|---|
| AUC Test | 0.7118 | — |
| Gini Test | 0.4237 | — |
| KS Test | 31.32% | — |
| PSI (Train vs Test) | 0.0001 | ✅ Ổn định |
| Bootstrap Gini 95% CI | [0.408, 0.438] | ✅ Không phải may mắn |
| Bootstrap KS 95% CI | [30.04%, 32.74%] | ✅ Ổn định |

---

## Risk Quantification — Monte Carlo + Stress Test

Áp dụng **Vasicek One-Factor Model** (nền tảng Basel II IRB) để định lượng rủi ro danh mục từ PD output của Scorecard.

### Công thức Vasicek

$$PD_i^{conditional} = \Phi\left(\frac{\Phi^{-1}(PD_i) - \sqrt{\rho} \cdot Z}{\sqrt{1-\rho}}\right)$$

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| ρ (rho) | 0.15 | Hệ số tương quan tài sản (retail lending) |
| LGD | Beta(α=4.5, β=5.5) | Stochastic, mean=45%, std=15% |
| EAD | AMT_CREDIT thực tế | Dư nợ tại thời điểm vỡ nợ |
| Z | ~N(0,1) | Cú shock kinh tế hệ thống |

### Monte Carlo (10,000 kịch bản)

| Metric | Giá trị |
|---|---|
| Expected Loss (EL) | 1.22B VND (EL/EAD: 3.32%) |
| VaR 95% | 2.95B VND |
| VaR 99% | 4.34B VND |
| Unexpected Loss (UL) | 3.12B VND |
| Capital Ratio (UL/EAD) | **8.49%** |

### Stress Test (6 kịch bản)

| Kịch bản | Z | PD TB | Tổn thất | Loss/EAD |
|---|---|---|---|---|
| Baseline | 0.0 | 6.65% | 0.99B VND | 2.70% |
| Mild Stress | -0.5 | 9.50% | 1.43B VND | 3.89% |
| Moderate Stress | -1.0 | 13.18% | 2.00B VND | 5.44% |
| Severe Stress | -1.5 | 17.74% | 2.72B VND | 7.39% |
| Extreme Stress | -2.0 | 23.19% | 3.58B VND | 9.74% |
| High Extreme | -2.5 | 29.48% | 4.59B VND | 12.48% |

---

## Pipeline 3 — E2E Production System

### Kiến trúc hệ thống

```
Browser (User)
    ↓ :8501
Streamlit Frontend
    ↓ HTTP :8000
FastAPI Backend
    POST /predict        — 1 hồ sơ JSON
    POST /predict_batch  — CSV upload, N records
    GET  /               — Health check
    ↓ joblib.load
5 × LightGBM Models (.pkl) + model_metadata.json
```

### Cấu trúc thư mục

```
E2E - Home Credit/
├── backend/
│   ├── main.py
│   ├── artifacts/
│   │   ├── lgb_fold_1.pkl ~ lgb_fold_5.pkl
│   │   ├── model_metadata.json
│   │   └── mock_samples.json
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── Notebooks/
├── docker-compose.yml
└── README.md
```

### Hướng dẫn chạy

**Local:**
```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

**Docker Compose:**
```bash
docker-compose up --build
```

Truy cập: [http://localhost:8501](http://localhost:8501) | API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Stack công nghệ

| Layer | Công nghệ |
|---|---|
| ML Pipeline | LightGBM, Optuna, SHAP |
| Scorecard Pipeline | statsmodels, scipy, scikit-learn |
| Risk Quantification | scipy.stats (Vasicek), NumPy Monte Carlo |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Data Processing | Pandas, NumPy |
| Model Validation | Pydantic |

---

## Ghi chú kỹ thuật

- **NaN handling**: LightGBM xử lý native — không impute; Scorecard tạo bin `_MISSING_` riêng với WoE tính từ train set
- **Data leakage prevention**: WoE bin edges fit chỉ trên train, transform lên cả train và test
- **Categorical features**: ép `dtype=category` trước LightGBM inference; WoE transform cho Logistic Regression
- **Ensemble inference**: trung bình cộng `predict_proba[:, 1]` từ 5 fold
- **Docker networking**: Streamlit đọc `BACKEND_URL` từ `os.getenv` — local dùng `127.0.0.1`, Docker dùng DNS nội bộ `http://backend:8000`
- **Vasicek convention**: công thức dùng dấu trừ trước `√ρ·Z` — Z âm tương ứng môi trường kinh tế xấu (PD tăng)
