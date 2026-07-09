import io
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import create_model
from typing import Any

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Home Credit Default Risk Inference API")

# 1. Tải cấu trúc Metadata và khởi tạo validation model động
METADATA_PATH = "artifacts/model_metadata.json"
try:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Không tìm thấy file cấu trúc tại {METADATA_PATH}")

feature_names = metadata["feature_names"]
categorical_features = metadata["categorical_features"]

# Tự động tạo cấu trúc Pydantic Model cho 133 biến độc lập từ file metadata
# Đặt giá trị mặc định là None (Any) vì dữ liệu tín dụng thô thường bị thiếu (NaN)
# Dùng LightGBM tự xử lý dữ liệu
input_fields = {col: (Any, None) for col in feature_names}
CustomerInputModel = create_model("CustomerInputModel", **input_fields)

# 2. Nạp 5 mô hình K-fold vào Ram
models = []
for fold in range(1, 6):
    model_path = f"artifacts/lgb_fold_{fold}.pkl"
    try:
        # Sử dụng joblib để nạp lại mô hình dạng cây tối ưu hơn pickle
        model = joblib.load(model_path)
        models.append(model)
    except Exception as e:
        raise RuntimeError(f"Lỗi khi nạp mô hình fold {fold}: {str(e)}")
print(f"Hệ thống: Đã nạp thành công {len(models)} mô hình K-Fold vào RAM.")


# 3. Định tuyến các Endpoint API
@app.get("/")
def index():
    """Endpoint kiểm tra trạng thái hoạt động của API"""
    return {"status": "healthy", "message": "API Chấm điểm tín dụng đang hoạt động."}


@app.post("/predict")
def predict(customer_data: CustomerInputModel):  # type: ignore
    """Endpoint tiếp nhận 133 biến của 1 khách hàng và trả về xác xuất vỡ nợ"""
    try:
        # Chuyển đổi dữ liệu đầu vào từ Pydantic sang Dictionary Python
        raw_dict = customer_data.dict()

        # Chuyển thành pd.DataFrame, ép đúng thứ tự lúc huấn luyện
        df = pd.DataFrame([raw_dict])

        # Duyệt qua tất cả cột của df
        for col in df.columns:
            # Kiểm tra nếu cột là cột định tính
            if col in categorical_features:
                # Ép kiểu về lại category
                df[col] = df[col].astype("category")
            # Trường hợp là cột định lượng
            else:
                # Ép sang số, đổi "None" thành "NaN" và đưa về float/int
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Tính trung bình cộng xác suất dự báo từ 5 fold
        probabilities = []
        for model in models:
            # Predict_proba trả về mảng [[prob_o, prob_1]] -> Lấy 1: xác suất vỡ nợ
            prob = model.predict_proba(df)[0, 1]
            probabilities.append(prob)

        # Tính giá trị trung bình cuối cùng
        final_probability = sum(probabilities) / len(probabilities)

        # Thiết lập ngưỡng cắt (threshold) mặc định là 0,5 để phân loại >= 0,5 nghĩa là vỡ nợ
        threshold = 0.5
        prediction = 1 if final_probability >= threshold else 0

        return {
            "credit_risk_probability": float(final_probability),
            "decision": "Reject" if prediction == 1 else "Approve",
            "threshold_rule": threshold,
            "fold_details": [float(p) for p in probabilities],
        }
    except Exception as e:
        # Trả về về lỗi 500 nếu ép kiểu hoặc dự báo lỗi
        raise HTTPException(status_code=500, detail=f"Lỗi xử lí dữ liệu: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Nhận file CSV, trả về kết quả"""
    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))

        # Nhận feature theo thứ tự
        df = df_raw[feature_names].copy()

        # Ép kiểu
        for col in df.columns:
            # Kiểm tra nếu cột là cột định tính
            if col in categorical_features:
                # Ép kiểu về lại category
                df[col] = df[col].astype("category")
            # Trường hợp là cột định lượng
            else:
                # Ép sang số, đổi "None" thành "NaN" và đưa về float/int
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensemble inference toàn batch
        all_probs = []
        for model in models:
            probs = model.predict_proba(df)[:, 1]
            all_probs.append(probs)

        final_probs = np.mean(all_probs, axis=0)
        threshold = 0.5
        decisions = ["Reject" if p >= threshold else "Approve" for p in final_probs]

        print(f"all_probs shape: {[len(p) for p in all_probs]}")
        print(
            f"final_probs type: {type(final_probs)}, shape: {np.array(final_probs).shape}"
        )

        return {
            "total_records": len(df),
            "results": [
                {
                    "record_index": i,
                    "credit_risk_probability": float(final_probs[i]),
                    "decision": decisions[i],
                }
                for i in range(len(df))
            ],
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Thiếu cột trong CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lí batch: {str(e)}")
