import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import DEVICE
import time
from YoloBackbone import YoloBackbone
from RViT import RViT

#--- YOLO_RViT ---
class YOLO_RViT(nn.Module):
    def __init__(self, yolo_path, yolo_target_feature_layer_idx=9,size=(640,640)):
        super().__init__()
        self.backbone = YoloBackbone(yolo_path, target_feature_layer_index=yolo_target_feature_layer_idx)
        dummy_input = torch.randn(1, 3, size[0], size[1]).to(DEVICE)
        
        original_backbone_training_mode = self.backbone.training
        with torch.no_grad():
            dummy_feats = self.backbone(dummy_input)
        
        yolo_channels = dummy_feats.shape[1]
        h_feat, w_feat = dummy_feats.shape[2], dummy_feats.shape[3]
        num_patches = h_feat * w_feat
        
        self.rvit = RViT(yolo_channels=yolo_channels, num_patches=num_patches).to(DEVICE)

    def forward(self, x, target=None, teach_ratio=0.5, forced_output_length=None):
        x = x.to(DEVICE)
        t1 = time.perf_counter()
        feats = self.backbone(x)
        t2 = time.perf_counter()
        print(f"Backbone time: {(t2 - t1)*1000:.2f} ms")
        output = self.rvit(feats, target, teach_ratio, forced_output_length)
        t3 = time.perf_counter()
        print(f"RViT time: {(t3 - t2)*1000:.2f} ms")
        return output

    def train(self, mode: bool = True):
        super().train(mode)
        self.rvit.train(mode)
        self.backbone.train(mode)
        return self

    def eval(self):
        super().eval()
        self.rvit.eval()
        self.backbone.eval()
        return self