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
import time
import numpy as np

MEAN_TENSOR = None
STD_TENSOR = None

def init_normalization_tensors():
    """Khởi tạo tensor normalization trên GPU một lần"""
    global MEAN_TENSOR, STD_TENSOR
    if MEAN_TENSOR is None:
        MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        STD_TENSOR = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# --- Hàm tiền xử lý trên GPU ---
def preprocess_image(frame, frame_size=(640, 640)):
    global MEAN_TENSOR, STD_TENSOR
    init_normalization_tensors()

    # Resize bằng OpenCV (CPU - vẫn nhanh)
    if frame.shape[:2] != frame_size:
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
    
     # Convert to tensor và chuyển lên GPU ngay
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0)  # NCHW
    
    # Chuyển lên GPU rồi normalize (GPU nhanh hơn CPU cho phép tính)
    tensor = tensor.to(DEVICE, non_blocking=True)
    tensor = (tensor - MEAN_TENSOR) / STD_TENSOR
    
    return tensor

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
    model.eval() # Chuyển model sang chế độ đánh giá 
    
    # Log kiểm tra model ở đâu
    print(f"[CHECK] Model device: {next(model.parameters()).device}")
    init_normalization_tensors()
    return model

# --- Hàm dự đoán ---
def predict_license_plate(model, frame, size=(640, 640), constrained_length=None):
    # Đo thời gian preprocess (CPU)
    t1 = time.perf_counter()
    image_tensor = preprocess_image(frame, size)
    t2 = time.perf_counter()
    
    # Đo thời gian chuyển sang GPU
    image_tensor = image_tensor.to(DEVICE)
    t3 = time.perf_counter()

     # Tự động chuyển sang FP16 nếu model đang dùng FP16
    if next(model.parameters()).dtype == torch.float16:
        image_tensor = image_tensor.half()
    
    print(f"[CHECK] Input tensor device: {image_tensor.device}")
    print(f"[CHECK] Model device: {next(model.parameters()).device}")
    print(f"[TIME] Preprocess (CPU): {(t2-t1)*1000:.2f}ms")
    print(f"[TIME] Transfer to GPU: {(t3-t2)*1000:.2f}ms")

    # Đo thời gian inference
    t4 = time.perf_counter()
    with torch.no_grad(): # Không cần tính gradient khi dự đoán
        outputs_logits = model(image_tensor, target=None, teach_ratio=0.0, forced_output_length=constrained_length)
    
    # Sync GPU để đo chính xác
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    t5 = time.perf_counter()
    print(f"[TIME] Model inference (GPU): {(t5-t4)*1000:.2f}ms")
    
    # Đo thời gian postprocess - TỐI ƯU HÓA
    t6 = time.perf_counter()
    
    # Softmax trên GPU
    probabilities = torch.softmax(outputs_logits, dim=-1)  # (batch, seq_len, num_classes)
    
    # Max trên GPU
    max_probs, predicted_indices_tensor = torch.max(probabilities, dim=-1)  # (batch, seq_len)
    
    # ✅ OPTIMIZATION 2: Chỉ lấy batch đầu tiên, nhưng vẫn trên GPU
    predicted_indices_gpu = predicted_indices_tensor[0]  # (seq_len,)
    confidence_per_char_gpu = max_probs[0]  # (seq_len,)
    
    # Tìm vị trí EOS token trên GPU (nhanh hơn)
    eos_mask = (predicted_indices_gpu == 36)  # EOS_TOKEN = 36
    if eos_mask.any():
        eos_pos = torch.nonzero(eos_mask, as_tuple=False)[0, 0].item()
        predicted_indices_gpu = predicted_indices_gpu[:eos_pos]
        confidence_per_char_gpu = confidence_per_char_gpu[:eos_pos]
    
    # Thay vì transfer từng cái một, transfer cả batch
    predicted_indices_list = predicted_indices_gpu.cpu().tolist()
    confidence_per_char = confidence_per_char_gpu.cpu().tolist()

    # Decode text (vẫn phải chạy trên CPU vì dùng Python string operations)
    predicted_text = index_to_char(predicted_indices_list, include_special_tokens=False)

    # Tính overall confidence
    if not confidence_per_char:
        overall_confidence = 0.0
    else:
        overall_confidence = sum(confidence_per_char) / len(confidence_per_char)

    t7 = time.perf_counter()
    print(f"[TIME] Postprocess (CPU): {(t7-t6)*1000:.2f}ms")
    print(f"[TIME] Total: {(t7-t1)*1000:.2f}ms")
    print("-" * 40)
    
    return predicted_text, overall_confidence
