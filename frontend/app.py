import streamlit as st
import requests
import json
import pandas as pd
import os

# Cấu hình tiêu đề và bố cục giao diện trang web
st.set_page_config(page_title="Home Credit Risk Scoring", layout="wide")

st.title("Hệ thống Chấm điểm Tín dụng & Dự báo Rủi ro Vỡ nợ")
st.caption("Môi trường Production Local - Sử dụng Mô hình Ensemble LightGBM K-Fold")


# SỬA LẠI ĐOẠN NÀY: Để hệ thống tự động lấy cấu hình từ Docker Compose truyền vào
# Nếu không chạy trong Docker, nó sẽ tự động dùng địa chỉ mặc định 127.0.0.1 bên dưới
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/predict")
BACKEND_BATCH_URL = os.getenv(
    "BACKEND_BATCH_URL", "http://127.0.0.1:8000/predict-batch"
)


# 1. Đọc dữ liệu Mock để hiển thị lên giao diện
# Để frontend đọc được, copy `mock_samples.json` qua folder frontend
@st.cache_data
def load_mock_data():
    try:
        # Đọc file mock
        with open("mock_samples.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(
            "Không tìm thấy file mock_samples.json. Hãy copy file này vào thư mục frontend."
        )
        return None


customers = load_mock_data()

if customers:
    # Tạo danh sách lựa chọn khách hàng (giao diện dropdown)
    customer_options = [f"Khách hàng ID: {idx + 1}" for idx in range(len(customers))]
    selected_customer_label = st.selectbox(
        "Chọn hồ sơ khách hàng để kiểm thử:", customer_options
    )

    # Lấy dữ liệu dictionary tương ứng với khách hàng được chọn
    customer_idx = customer_options.index(selected_customer_label)
    current_customer_data = customers[customer_idx].copy()
    # Tách TARGET khỏi mock_samples, sau khi ra kết quả thì gán nó vào để đối chiếu
    actual_target = current_customer_data.pop("TARGET", None)

    st.write("Thay đổi biến quan trọng")
    st.info("Các biến độc lập có ảnh hưởng lớn, điều chỉnh thử nghiệm: ")

    # Tạo 3 cột để sửa thử nghiệm
    col1, col2, col3 = st.columns(3)
    with col1:
        ext_1 = st.number_input(
            "EXT_SOURCE_1",
            value=float(current_customer_data.get("EXT_SOURCE_1", 0.0) or 0.0),
            format="%.4f",
        )
    with col2:
        ext_2 = st.number_input(
            "EXT_SOURCE_2",
            value=float(current_customer_data.get("EXT_SOURCE_2", 0.0) or 0.0),
            format="%.4f",
        )
    with col3:
        ext_3 = st.number_input(
            "EXT_SOURCE_3",
            value=float(current_customer_data.get("EXT_SOURCE_3", 0.0) or 0.0),
            format="%.4f",
        )

    # Cập nhật lại giá trị được người dùng nhập
    current_customer_data["EXT_SOURCE_1"] = ext_1
    current_customer_data["EXT_SOURCE_2"] = ext_2
    current_customer_data["EXT_SOURCE_3"] = ext_3

    # Button kích hoạt gửi dữ liệu về backend
    if st.button("Chấm điểm tín dụng", type="primary"):
        with st.spinner("Chuyển dữ liệu xuống Backend và tính toán ..."):
            try:
                # Dùng thư viện toán học để kiểm tra NaN
                import math

                # Kiểm tra NaN, chuyển thành None, Json sẽ là Null
                cleaned_customer_data = {
                    k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in current_customer_data.items()
                }

                # Gửi request dạng Json xuống FastAPI
                response = requests.post(BACKEND_URL, json=cleaned_customer_data)

                if response.status_code == 200:
                    result = response.json()

                    # Phân tách giao diện do Backend trả về
                    st.write("---")
                    st.write("KẾT QUẢ KIỂM TRA")

                    prob = result["credit_risk_probability"]
                    decision = result["decision"]

                    # Hiển thị Badge màu sắc theo quyết định duyệt
                    if decision == "Approve":
                        st.success(f"**DUYỆT**")
                    else:
                        st.error(f"**TỪ CHỐI**")

                    # Hiển thị các thông số chi tiết
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Tỉ lệ vỡ nợ", value=f"{prob * 100:.2f}%")
                    with metric_col2:
                        st.metric(
                            label="Ngưỡng rủi ro",
                            value=f"{result['threshold_rule'] * 100: .1f}%",
                        )

                    # So sánh kết quả dự đoán và TARGET
                    if actual_target is not None:
                        st.write("**So sánh với thực tế đã xảy ra TARGET**")
                        gt_col1, gt_col2 = st.columns(2)
                        with gt_col1:
                            label_text = (
                                "Vỡ nợ (1)" if actual_target == 1 else "Không vỡ nợ (0)"
                            )
                            st.metric(label="Thực tế", value=label_text)
                        with gt_col2:
                            is_correct = (actual_target == 1) == (decision == "Reject")
                            st.metric(
                                label="Dự báo", value="Đúng" if is_correct else "Sai"
                            )

                    # Chi tiết từng fold
                    st.write("**Chi tiết 5 folds**")
                    fold_df = pd.DataFrame(
                        {
                            f"Fold {i+1}": [f"{p*100:.2f}%"]
                            for i, p in enumerate(result["fold_details"])
                        }
                    )
                    st.table(fold_df)
                else:
                    st.error(f"Lỗi hệ thống: {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error(
                    "Không thể kết nối. Hãy chắc chắn rằng bạn đã chạy lệnh `uvicorn main:app` ở cổng 8000."
                )

st.write("---")
st.write("**Batch Prediction từ CSV**")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    if st.button("Chấm điểm batch", type="primary"):
        with st.spinner("Đang xử lý records..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict_batch",
                    files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                )
                if response.status_code == 200:
                    result = response.json()
                    df_result = pd.DataFrame(result["results"])
                    st.success(f"Hoàn thành: {result['total_records']} records")
                    st.dataframe(df_result)
                else:
                    st.error(f"Lỗi: {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error("Không thể kết nối backend.")
