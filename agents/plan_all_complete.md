# Kế hoạch chi tiết - Dự đoán giá nhà California (Lab 04)

Tài liệu này trình bày kế hoạch các bước triển khai chi tiết cho bài toán dự báo giá nhà.

---

## Các giai đoạn thực hiện

### 1. Phân tích khám phá dữ liệu (EDA)
- **Đọc và Tổng quan**: Xem thông tin tổng quát về kích thước dữ liệu, kiểu dữ liệu (`df.info()`), các thuộc tính thống kê cơ bản (`df.describe()`).
- **Phân phối của nhãn**: Xem phân phối giá trị nhà trung vị `median_house_value`, xác định giới hạn trần nếu có (ở mức $500,001).
- **Phân tích địa lý**: Sử dụng các biểu đồ phân tán kinh độ/vĩ độ kết hợp với quy mô dân số (s) và giá trị nhà (c) để phát hiện xu hướng giá cao tập trung ở vùng ven biển.
- **Tương quan đặc trưng**: Tính ma trận tương quan Pearson, vẽ heatmap để phát hiện mối quan hệ mạnh nhất (ví dụ: thu nhập trung vị `median_income` có độ tương quan rất cao với giá nhà).
- **Giá trị khuyết thiếu**: Thống kê số lượng giá trị NaN ở thuộc tính `total_bedrooms` (chiếm khoảng 1%).

### 2. Tiền xử lý dữ liệu (Data Preprocessing)
- **Tách tập dữ liệu**: Phân chia tập huấn luyện (train) và kiểm thử (test) phân tầng theo phân khúc thu nhập (`StratifiedShuffleSplit`) để bảo toàn tính phân phối của dữ liệu gốc.
- **Làm sạch dữ liệu**: Sử dụng Imputer điền khuyết các giá trị thiếu trong `total_bedrooms` bằng giá trị trung vị (median).
- **Xử lý thuộc tính phân loại**: Mã hóa One-Hot cho thuộc tính `ocean_proximity`.
- **Tạo đặc trưng phái sinh**: Thêm các thuộc tính tỷ lệ `rooms_per_household`, `population_per_household`, `bedrooms_per_room`.
- **Chuẩn hóa quy mô**: Áp dụng chuẩn hóa z-score (`StandardScaler`) cho các đặc trưng đầu vào và chuẩn hóa biến mục tiêu để mô hình mạng nơ-ron học ổn định hơn.

### 3. Huấn luyện mô hình MLP tự viết từ đầu (model.py)
- **Kiến trúc**: Mạng nơ-ron truyền thẳng (Feedforward Neural Network) với:
  - Lớp ẩn 1: 64 nơ-ron, hàm kích hoạt ReLU.
  - Lớp ẩn 2: 32 nơ-ron, hàm kích hoạt ReLU.
  - Lớp đầu ra: 1 nơ-ron (dự đoán giá trị liên tục).
- **Thuật toán tối ưu**: Lan truyền ngược (Backpropagation) và cập nhật trọng số bằng Gradient Descent cơ bản với tốc độ học `lr = 0.1`.
- **Huấn luyện**: Thiết lập số epoch là `5000`. Theo dõi sự giảm dần của hàm mất mát Mean Squared Error (MSE).

### 4. Đánh giá hiệu năng và Kết xuất
- **Dự đoán và Chuyển ngược**: Sử dụng mô hình đã huấn luyện để dự đoán trên tập test, chuyển ngược nhãn về đơn vị gốc (USD) thông qua `inverse_transform` của scaler.
- **Chỉ số đánh giá**: Tính toán độ lỗi Root Mean Squared Error (RMSE) và Mean Absolute Error (MAE) trên đơn vị thực tế.
- **Kết xuất**: Lưu trữ dữ liệu preprocessed vào `data/processed/housing_prepared.csv`.
