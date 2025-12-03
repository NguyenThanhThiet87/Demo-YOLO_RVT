"""
Phiên bản tối ưu của PredictModel.py
Tối ưu hóa các operations có thể cải thiện ngay:
1. Optimize postprocessing - giữ operations trên GPU lâu hơn
2. Batch operations trước khi transfer về CPU
3. Reduce unnecessary CPU-GPU transfers
"""

import torch
import torch.nn.functional as F
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


# --- Hàm tiền xử lý (giữ nguyên hoặc có thể optimize thêm) ---
def preprocess_image(frame, frame_size=(640, 640)):
    """
    Preprocess image - giữ nguyên vì resize trên CPU thường đủ nhanh
    Nếu muốn optimize thêm, có thể dùng torchvision transforms trên GPU
    """
    global MEAN_TENSOR, STD_TENSOR
    init_normalization_tensors()

    # Đưa dữ liệu sang tensor và giữ bộ nhớ liên tục để transfer nhanh hơn
    frame_tensor = torch.from_numpy(np.ascontiguousarray(frame))  # HWC, uint8
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # -> NCHW
    
    # Giữ bộ nhớ pinned để copy H2D nhanh, sau đó chuyển GPU và scale về [0,1]
    if frame_tensor.device.type == 'cpu':
        frame_tensor = frame_tensor.pin_memory()
    frame_tensor = frame_tensor.to(DEVICE, non_blocking=True, dtype=torch.float32)
    frame_tensor.mul_(1.0 / 255.0)
    
    # Resize trực tiếp trên GPU để tận dụng CUDA
    if frame_tensor.shape[-2:] != frame_size:
        frame_tensor = F.interpolate(frame_tensor, size=frame_size, mode='bilinear', align_corners=False)
    
    # Normalize trên GPU
    frame_tensor = (frame_tensor - MEAN_TENSOR) / STD_TENSOR
    
    return frame_tensor


# --- Hàm tải model và trọng số (giữ nguyên) ---
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


# --- Hàm dự đoán TỐI ƯU ---
def predict_license_plate(model, frame, size=(640, 640), constrained_length=None):
    """
    Phiên bản tối ưu của predict_license_plate
    Các cải thiện:
    1. Giữ tất cả operations trên GPU càng lâu càng tốt
    2. Batch các operations trước khi transfer về CPU
    3. Giảm số lần GPU-CPU synchronization
    """
    # Đo thời gian preprocess (CPU)
    t1 = time.perf_counter()
    image_tensor = preprocess_image(frame, size)
    t2 = time.perf_counter()
    
    # Đo thời gian chuyển sang GPU (tensor đã nằm trên GPU nên thời gian gần như 0)
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
    
    # ✅ OPTIMIZATION 1: Giữ tất cả operations trên GPU
    # Thay vì:
    #   probalities = torch.softmax(...)
    #   max_probs, indices = torch.max(...)
    #   indices_list = indices[0].cpu().tolist()  # Transfer ngay
    #   
    # Ta làm:
    #   Tất cả processing trên GPU, chỉ transfer 1 lần ở cuối
    
    # Softmax trên GPU
    probabilities = torch.softmax(outputs_logits, dim=-1)  # (batch, seq_len, num_classes)
    
    # Max trên GPU
    max_probs, predicted_indices_tensor = torch.max(probabilities, dim=-1)  # (batch, seq_len)
    
    # ✅ OPTIMIZATION 2: Chỉ lấy batch đầu tiên, nhưng vẫn trên GPU
    predicted_indices_gpu = predicted_indices_tensor[0]  # (seq_len,)
    confidence_per_char_gpu = max_probs[0]  # (seq_len,)
    
    # ✅ OPTIMIZATION 3: Tìm EOS token trên GPU trước khi transfer
    # Tìm vị trí EOS token trên GPU (nhanh hơn)
    eos_mask = (predicted_indices_gpu == 36)  # EOS_TOKEN = 36
    if eos_mask.any():
        eos_pos = torch.nonzero(eos_mask, as_tuple=False)[0, 0].item()
        predicted_indices_gpu = predicted_indices_gpu[:eos_pos]
        confidence_per_char_gpu = confidence_per_char_gpu[:eos_pos]
    
    # ✅ OPTIMIZATION 4: Transfer tất cả về CPU cùng 1 lúc (hiệu quả hơn)
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


def predict_license_plate_batch(model, frames, size=(640, 640), constrained_length=None):
    """
    Phiên bản batch processing - xử lý nhiều frames cùng lúc
    Tận dụng parallel processing của GPU tốt hơn
    
    Args:
        model: Model đã load
        frames: List of frames (numpy arrays)
        size: Target size
        constrained_length: Optional length constraint
    
    Returns:
        List of (predicted_text, overall_confidence) tuples
    """
    if not frames:
        return []
    
    # Preprocess tất cả frames
    batch_tensors = []
    for frame in frames:
        tensor = preprocess_image(frame, size)
        batch_tensors.append(tensor)
    
    # Stack thành batch
    batch_input = torch.cat(batch_tensors, dim=0)  # (batch_size, 3, 640, 640)
    
    if next(model.parameters()).dtype == torch.float16:
        batch_input = batch_input.half()
    
    # Inference batch
    with torch.no_grad():
        outputs_logits = model(batch_input, target=None, teach_ratio=0.0, forced_output_length=constrained_length)
    
    # Postprocess batch
    probabilities = torch.softmax(outputs_logits, dim=-1)  # (batch_size, seq_len, num_classes)
    max_probs, predicted_indices_tensor = torch.max(probabilities, dim=-1)  # (batch_size, seq_len)
    
    results = []
    for i in range(batch_input.size(0)):
        pred_indices = predicted_indices_tensor[i]
        conf_per_char = max_probs[i]
        
        # Tìm EOS trên GPU
        eos_mask = (pred_indices == 36)  # EOS_TOKEN
        if eos_mask.any():
            eos_pos = torch.nonzero(eos_mask, as_tuple=False)[0, 0].item()
            pred_indices = pred_indices[:eos_pos]
            conf_per_char = conf_per_char[:eos_pos]
        
        # Transfer về CPU
        pred_list = pred_indices.cpu().tolist()
        conf_list = conf_per_char.cpu().tolist()
        
        # Decode
        predicted_text = index_to_char(pred_list, include_special_tokens=False)
        overall_confidence = sum(conf_list) / len(conf_list) if conf_list else 0.0
        
        results.append((predicted_text, overall_confidence))
    
    return results


