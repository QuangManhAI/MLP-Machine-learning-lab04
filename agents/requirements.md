# Requirements - Dự đoán giá nhà California (Lab 04)

Tài liệu này tóm tắt yêu cầu, dữ liệu đầu vào, đầu ra và các bước triển khai của bài toán dự đoán giá nhà trong bài thực hành Lab 04.

---

## 1. Purpose - Mục tiêu
Xây dựng hệ thống dự báo giá nhà trung vị (`median_house_value`) tại bang California dựa trên các thông tin nhân khẩu học và địa lý.
Yêu cầu đặc biệt là tự viết thuật toán Multi-Layer Perceptron (MLP) từ đầu (sử dụng NumPy) để huấn luyện và đối chiếu hiệu năng.

---

## 2. Input - Dữ liệu đầu vào
* File dữ liệu: `data/raw/housing.csv`
* Các đặc trưng chính: Kinh độ (`longitude`), Vĩ độ (`latitude`), Tuổi nhà trung vị (`housing_median_age`), Tổng số phòng (`total_rooms`), Tổng số phòng ngủ (`total_bedrooms`), Dân số (`population`), Số hộ gia đình (`households`), Thu nhập trung vị (`median_income`), Biến phân loại vị trí gần biển (`ocean_proximity`).

---

## 3. Output - Kết quả đầu ra
1. **Thư viện mã nguồn dùng chung trong src/lab04/**:
   - `eda.py`: Chứa các hàm vẽ biểu đồ phân tích và trực quan hóa phân phối giá nhà, tương quan đặc trưng, v.v.
   - `process.py`: Chứa pipeline làm sạch dữ liệu, mã hóa và thêm đặc trưng phái sinh (`CombinedAttributesAdder`).
   - `model.py`: Chứa lớp mô hình mạng nơ-ron tự viết `MultiLayerPercepTron` từ đầu sử dụng NumPy.
2. **Notebook chạy hoàn chỉnh**:
   - `notebooks/lab04.ipynb`: Chạy thông suốt từ khâu nạp dữ liệu, gọi các hàm EDA từ gói `lab04`, tiền xử lý, huấn luyện MLP tự xây dựng, và in ra các chỉ số RMSE, MAE.

---

## 4. How to do - Các bước thực hiện

### Bước 1: Khám phá dữ liệu (EDA)
* Sử dụng các hàm trực quan trong `lab04.eda` để vẽ bản đồ mật độ địa lý, biểu đồ phân phối giá nhà, ma trận tương quan heatmap.

### Bước 2: Tiền xử lý dữ liệu - Feature Engineering
* Sử dụng `CombinedAttributesAdder` để tạo thêm các thuộc tính tỷ lệ phòng ngủ, phòng trên hộ gia đình.
* Thiết lập `ColumnTransformer` để tự động hóa quy trình điền dữ liệu thiếu, mã hóa One-Hot biến phân loại, và chuẩn hóa StandardScaler.

### Bước 3: Huấn luyện và Đánh giá
* Sử dụng lớp `MultiLayerPercepTron` tự viết để fit trên tập huấn luyện đã chuẩn hóa.
* Dự đoán trên tập test, chuyển ngược thang đo giá nhà (inverse transform) về đơn vị thực tế (USD) và tính toán RMSE, MAE.
