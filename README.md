# California Housing Price Prediction (Lab 04)

Dự án này là bài thực hành Lab 04 về xây dựng mô hình mạng nơ-ron Multi-Layer Perceptron (MLP) tự viết từ đầu bằng NumPy để dự đoán giá nhà trung vị tại bang California.

---

## 1. Cấu trúc thư mục dự án

```text
MLP-Machine-learning-lab04/
├── agents/                      # Thư mục chứa yêu cầu, tài liệu mô tả cho dự án
│   ├── description.md           # Hướng dẫn cấu trúc notebook và yêu cầu refactor cụ thể
│   ├── requirements.md          # Yêu cầu tổng quan của Lab 04 (Mục tiêu, Input, Output)
│   └── plan_all_complete.md     # Kế hoạch chi tiết để dự đoán giá nhà
├── data/                        # Thư mục chứa dữ liệu
│   ├── raw/                     # Dữ liệu nguồn chưa qua xử lý (housing.csv)
│   └── processed/               # Dữ liệu sau khi qua tiền xử lý (housing_prepared.csv)
├── notebooks/                   # Jupyter Notebooks thực hiện phân tích và mô hình
│   ├── 02_end_to_end_machine_learning_project.ipynb  # Notebook mẫu tham khảo
│   └── lab04.ipynb              # Notebook chính chứa toàn bộ pipeline và huấn luyện mô hình
├── src/                         # Mã nguồn chính của dự án (được đóng gói thành thư viện dùng chung)
│   └── lab04/
│       ├── __init__.py
│       ├── eda.py               # Các hàm EDA vẽ biểu đồ phân tích dữ liệu
│       ├── process.py           # Pipeline xử lý dữ liệu và tạo đặc trưng phái sinh
│       └── model.py             # Lớp mạng nơ-ron MLP Regressor tự viết từ đầu bằng NumPy
├── artifacts/                   # Lưu trữ các file mô hình huấn luyện xong
├── reports/                     # Lưu trữ các báo cáo kết quả và hình ảnh biểu đồ EDA
├── scripts/                     # Các file script bổ trợ chạy độc lập
├── pyproject.toml               # File cấu hình cài đặt package lab04 ở chế độ editable (pip install -e .)
├── requirements.txt             # Các thư viện Python cần thiết
└── README.md                    # Hướng dẫn tổng quan về dự án
```

---

## 2. Hướng dẫn cài đặt và sử dụng

### Cài đặt môi trường ảo
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy Jupyter Lab / Notebook
```bash
jupyter notebook notebooks/lab04.ipynb
```