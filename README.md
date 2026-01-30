YOLO-RVT: End-to-End License Plate Recognition
📌 Giới thiệu

YOLO-RVT là một framework End-to-End License Plate Recognition (LPR) sử dụng sự kết hợp giữa YOLO và Recurrent Vision Transformer (RVT) nhằm nhận dạng biển số xe một cách chính xác và hiệu quả.

Khác với các hệ thống LPR hai giai đoạn truyền thống (phát hiện → nhận dạng), YOLO-RVT giải quyết các vấn đề như:

Lan truyền lỗi giữa các giai đoạn

Khó khăn khi xử lý chuỗi ký tự có độ dài thay đổi

Điều kiện môi trường phức tạp (ánh sáng yếu, mờ, góc nhìn nghiêng)

Framework này đặc biệt phù hợp cho Intelligent Transportation Systems (ITS).

🚀 Đặc điểm chính

✅ End-to-End Learning: Phát hiện và nhận dạng biển số trong một pipeline duy nhất

✅ YOLO Backbone: Trích xuất đặc trưng không gian hiệu quả, tốc độ cao

✅ Recurrent Vision Transformer (RVT): Mô hình hóa chuỗi ký tự với khả năng ghi nhớ ngữ cảnh

✅ Xử lý chuỗi ký tự độ dài thay đổi mà không cần phân đoạn ký tự thủ công

✅ Độ chính xác cao trong điều kiện ảnh khó (blur, ánh sáng kém, nhiều định dạng biển số)

🧠 Kiến trúc mô hình

YOLO-RVT gồm 2 thành phần chính:

YOLO Feature Extractor

Đóng vai trò backbone

Trích xuất feature map không gian từ ảnh đầu vào

Có thể huấn luyện end-to-end

Recurrent Vision Transformer (RVT)

Nhận feature map từ YOLO

Kết hợp cơ chế attention và recurrence

Dự đoán chuỗi ký tự biển số theo thứ tự

Input Image
     ↓
YOLO Backbone
     ↓
Feature Map
     ↓
Recurrent Vision Transformer
     ↓
License Plate Sequence Output

📈 Kết quả
<img width="1376" height="768" alt="Anh_da_loai_bo_Zalo_1" src="https://github.com/user-attachments/assets/215c4a8a-c9b5-44cc-85f5-be6e21499a53" />

