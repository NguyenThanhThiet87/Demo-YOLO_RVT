import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import index_to_char, char_to_indices, DEVICE
from YOLO_RViT import YOLO_RViT

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(frame, frame_size=(640, 640)):
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Chuyển đổi từ OpenCV (BGR) sang PIL (RGB)
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(frame)
    return img_tensor.unsqueeze(0) # Thêm batch dimension

# --- Hàm tải model và trọng số ---
def load_model_for_prediction(checkpoint_path, yolo_base_model_path, size=(640,640)):
    model = YOLO_RViT(
        yolo_path=yolo_base_model_path,
        yolo_target_feature_layer_idx=13
    )
    # Tải state_dict
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Thông tin từ checkpoint: {checkpoint_path}")
    print("-" * 40)
    print(f"Validation Character Accuracy (CRR): {checkpoint['val_acc_history'][-1]}")
    print(f"Validation Exact Match Accuracy (E2E RR): {checkpoint['val_acc_constrained_history'][-1]}")

    model.to(DEVICE)
    model.eval() # Chuyển model sang chế độ đánh giá (quan trọng!)
    return model

# --- Hàm dự đoán ---
def predict_license_plate(model, frame, size=(640, 640), constrained_length=None):
    image_tensor = preprocess_image(frame, size).to(DEVICE)

     # Tự động chuyển sang FP16 nếu model đang dùng FP16
    if next(model.parameters()).dtype == torch.float16:
        image_tensor = image_tensor.half()

    with torch.no_grad(): # Không cần tính gradient khi dự đoán
        outputs_logits = model(image_tensor, target=None, teach_ratio=0.0, forced_output_length=constrained_length)
    if(outputs_logits is None):
        return None, 0.0, 0.0, None
    probalities = torch.softmax(outputs_logits, dim=-1)  # Chuyển đổi logits thành xác suất
    
    max_probs, predicted_indices_tensor = torch.max(probalities, dim=-1)  # Lấy chỉ số của xác suất cao nhất
    predicted_indices_list = predicted_indices_tensor[0].cpu().tolist()  # Chuyển đổi tensor sang danh sách
    confidence_per_char = max_probs[0].cpu().tolist()  # Lấy xác suất cao nhất cho mỗi ký tự

    predicted_text = index_to_char(predicted_indices_list, include_special_tokens=False)

    if not confidence_per_char:
        overall_confidence = 0.0
    else:
        overall_confidence = sum(confidence_per_char) / len(confidence_per_char)

    # #Phân tích featureMap
    # max_activation = torch.max(featureMap).item()  # Lấy giá trị lớn nhất trong featureMap

    return predicted_text, overall_confidence
