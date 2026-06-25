# Yêu cầu dự án Lab 04 - Dự đoán giá nhà California

Tóm tắt các yêu cầu, dữ liệu đầu vào, kết quả cần đạt được và các bước thực hiện của bài Lab 04.

## 1. Mục tiêu
Xây dựng mô hình dự báo giá nhà ở bang California dựa trên dữ liệu nhân khẩu học và vị trí địa lý.
Yêu cầu quan trọng là tự cài đặt thuật toán mạng nơ-ron MLP từ đầu bằng thư viện NumPy, sau đó huấn luyện và đánh giá kết quả thay vì dùng các thư viện có sẵn như Scikit-Learn hay PyTorch cho phần mô hình.

## 2. Dữ liệu đầu vào
- File dữ liệu gốc đặt tại đường dẫn data/raw/housing.csv.
- Các thông tin đi kèm gồm tọa độ kinh độ, vĩ độ, tuổi của nhà, tổng số phòng, tổng số phòng ngủ, dân số, số hộ gia đình, mức thu nhập trung bình và khoảng cách tới biển.

## 3. Kết quả cần đạt được

### Các file mã nguồn trong thư mục src/lab04
- eda.py chứa các hàm vẽ biểu đồ phân phối giá nhà, bản đồ địa lý và ma trận tương quan.
- process.py chứa các hàm làm sạch dữ liệu, điền giá trị thiếu, mã hóa đặc trưng và tạo thuộc tính mới.
- model.py định nghĩa lớp MultiLayerPerceptron tự code bằng NumPy.

### Notebook chạy chính
- notebooks/lab04.ipynb dùng để chạy toàn bộ quy trình từ tải dữ liệu, chạy các bước phân tích, tiền xử lý, huấn luyện mô hình MLP tự viết và cuối cùng là tính toán các độ lỗi RMSE, MAE trên tập kiểm thử.

## 4. Các bước triển khai

### Bước 1: Phân tích dữ liệu
- Chạy các hàm vẽ bản đồ địa lý, biểu đồ phân phối và ma trận tương quan để hiểu rõ dữ liệu đầu vào.

### Bước 2: Chuẩn bị dữ liệu và tạo đặc trưng mới
- Tạo thêm các đặc trưng tỷ lệ như số phòng ngủ hoặc số phòng trên mỗi hộ.
- Viết pipeline để tự động hóa việc điền dữ liệu thiếu ở phần số phòng ngủ, mã hóa One-Hot cho dữ liệu dạng chữ và chuẩn hóa tất cả các đặc trưng về cùng một thang đo.

### Bước 3: Huấn luyện và đánh giá
- Sử dụng mô hình MLP tự viết để khớp dữ liệu trên tập huấn luyện.
- Dự đoán giá nhà trên tập kiểm thử, đưa kết quả về lại đơn vị USD thực tế rồi tính sai số RMSE và MAE để đánh giá mô hình.
