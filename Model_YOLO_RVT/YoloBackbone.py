import torch
import torch.nn as nn
from ultralytics import YOLO

from utils import DEVICE

#--- YOLO backbone ---
class YoloBackbone(nn.Module):
    def __init__(self, model_path, target_feature_layer_index=9):
        super().__init__()
        _temp_yolo_instance = YOLO(model_path)
        self.yolo_detection_model = _temp_yolo_instance.model
        self.yolo_detection_model.to(DEVICE)
        self.target_feature_layer_index = target_feature_layer_index

        for name, param in self.yolo_detection_model.named_parameters():
            param.requires_grad = True
        
        self._hook_handle = None
        self._fmap_out_hook = []
        
        self._register_hook()

    def _hook_fn_extractor(self, module, input_val, output_val):
        if isinstance(output_val, torch.Tensor):
            self._fmap_out_hook.append(output_val)
        elif isinstance(output_val, (list, tuple)):
            for item in output_val:
                if isinstance(item, torch.Tensor):
                    self._fmap_out_hook.append(item)
                    break

    def _register_hook(self):
        layer_to_hook = self.yolo_detection_model.model[self.target_feature_layer_index]
        self._hook_handle = layer_to_hook.register_forward_hook(self._hook_fn_extractor)

    def _remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, x):
        self._fmap_out_hook.clear()
        _ = self.yolo_detection_model(x)
        out_tensor = self._fmap_out_hook[0]
        return out_tensor if out_tensor.dim() == 4 else out_tensor.unsqueeze(0)