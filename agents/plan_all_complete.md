# Kế hoạch dự án Lab 04 - Dự báo giá nhà ở California
# https://www.kaggle.com/datasets/nazishjaveed/california-house-price-prediction
Kế hoạch triển khai các bước chạy mô hình dự báo giá nhà.

## Các đầu mục công việc chính

### 1. Khám phá dữ liệu ban đầu
- Xem thông tin tổng quan về dữ liệu, kiểu của các cột và các chỉ số thống kê cơ bản.
- Vẽ phân bố giá trị nhà trung bình để xem dải dữ liệu và phát hiện xem có bị chặn trần hay không.
- Vẽ đồ thị tọa độ kinh độ vĩ độ kết hợp với mật độ dân số và giá nhà để tìm vùng có giá nhà cao.
- Tính tương quan Pearson giữa các đặc trưng để tìm xem yếu tố nào ảnh hưởng nhiều nhất đến giá nhà.
- Thống kê tỷ lệ dữ liệu bị thiếu ở cột số phòng ngủ.

### 2. Tiền xử lý dữ liệu
- Chia dữ liệu thành hai tập train và test dựa theo phân lớp thu nhập để tránh bị lệch phân phối.
- Điền các giá trị thiếu ở cột số phòng ngủ bằng giá trị trung vị.
- Mã hóa One-Hot cho cột ocean_proximity là cột phân loại.
- Tạo thêm một số thuộc tính mới như số phòng trung bình mỗi hộ, dân số trung bình mỗi hộ và tỷ lệ phòng ngủ trên tổng số phòng.
- Chuẩn hóa dữ liệu đầu vào và đầu ra bằng StandardScaler để mô hình dễ hội tụ hơn.

### 3. Thiết kế và huấn luyện mô hình MLP tự code
- Mạng nơ-ron cơ bản gồm hai lớp ẩn với số node lần lượt là 64 và 32, sử dụng hàm kích hoạt ReLU.
- Lớp đầu ra có một node để dự báo giá nhà.
- Tự viết thuật toán lan truyền ngược Backpropagation và cập nhật trọng số bằng Gradient Descent với learning rate là 0.1.
- Huấn luyện mô hình qua 5000 vòng lặp và theo dõi hàm mất mát MSE giảm dần.

### 4. Đánh giá kết quả
- Dùng mô hình đã huấn luyện để dự đoán trên tập test, sau đó đưa giá trị dự đoán về đơn vị USD thực tế.
- Tính toán hai chỉ số đánh giá là RMSE và MAE.
- Lưu lại bộ dữ liệu sau khi đã tiền xử lý vào thư mục data.
