import torch
import torch.nn as nn
from ultralytics import YOLO

from utils import DEVICE

class EarlyExitException(Exception):
    """Exception để dừng forward pass sớm"""
    pass

class YoloBackbone(nn.Module):
    """
    YOLO Backbone tối ưu với Early Exit
    - Dừng forward ngay sau khi lấy được feature map
    - Không chạy detection head -> tiết kiệm ~30-40% thời gian
    - Tương thích hoàn toàn với checkpoint cũ
    """
    def __init__(self, model_path, target_feature_layer_index=13):
        super().__init__()
        _temp_yolo_instance = YOLO(model_path)
        self.yolo_detection_model = _temp_yolo_instance.model
        self.yolo_detection_model.to(DEVICE)
        self.target_feature_layer_index = target_feature_layer_index

        for name, param in self.yolo_detection_model.named_parameters():
            param.requires_grad = True
        
        # Feature map output
        self._fmap_out = None
        self._use_early_exit = True  # Flag để bật/tắt early exit
        
        self._register_hook()

    def _hook_fn_extractor(self, module, input_val, output_val):
        """Hook function - lưu feature map và raise exception để dừng sớm"""
        if isinstance(output_val, torch.Tensor):
            self._fmap_out = output_val
        elif isinstance(output_val, (list, tuple)):
            for item in output_val:
                if isinstance(item, torch.Tensor):
                    self._fmap_out = item
                    break
        
        # Early exit - dừng forward pass ngay tại đây
        if self._use_early_exit and not self.training:
            raise EarlyExitException()

    def _register_hook(self):
        layer_to_hook = self.yolo_detection_model.model[self.target_feature_layer_index]
        self._hook_handle = layer_to_hook.register_forward_hook(self._hook_fn_extractor)

    def _remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, x):
        self._fmap_out = None
        
        try:
            _ = self.yolo_detection_model(x)
        except EarlyExitException:
            # Đây là expected behavior khi early exit
            pass
        
        if self._fmap_out is None:
            raise RuntimeError("Failed to extract feature map from YOLO backbone")
        
        out_tensor = self._fmap_out
        return out_tensor if out_tensor.dim() == 4 else out_tensor.unsqueeze(0)
    
    def train(self, mode=True):
        """Override train để tắt early exit khi training"""
        super().train(mode)
        # Tắt early exit khi training (cần full forward cho gradient)
        self._use_early_exit = not mode
        return self
    
    def eval(self):
        """Override eval để bật early exit khi inference"""
        super().eval()
        self._use_early_exit = True
        return self