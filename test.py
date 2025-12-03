import torch
import torch.nn as nn
from ultralytics import YOLO

from Model_YOLO_RVT.utils import DEVICE


class YoloBackbone(nn.Module):
    """
    YOLO Backbone tối ưu hoàn toàn:
    - Cắt Detection Head (Layer 14-23)
    - Không dùng hook, không dùng exception
    - Forward thủ công xử lý skip connections
    - Tương thích 100% với checkpoint cũ
    - Scale tốt với GPU mạnh hơn
    """
    
    def __init__(self, model_path, target_feature_layer_index=13):
        super().__init__()
        
        # Load YOLO model
        _temp_yolo = YOLO(model_path)
        full_sequential = _temp_yolo.model.model  # nn.Sequential của tất cả layers
        
        self.target_feature_layer_index = target_feature_layer_index
        
        # Chỉ lấy layers từ 0 đến target (bao gồm target)
        # Layer 14-23 (Detection Head) sẽ bị loại bỏ hoàn toàn
        self.layers = nn.ModuleList()
        self.layer_from_info = []
        
        for i in range(target_feature_layer_index + 1):
            self.layers.append(full_sequential[i])
            self.layer_from_info.append(getattr(full_sequential[i], 'f', -1))
        
        # Xóa reference đến full model để giải phóng memory
        del _temp_yolo
        del full_sequential
        
        self.to(DEVICE)
        
        # Cho phép training
        for param in self.parameters():
            param.requires_grad = True
        
        print(f"[YoloBackbone] Extracted layers 0-{target_feature_layer_index}")
        print(f"[YoloBackbone] Detection Head (layers {target_feature_layer_index+1}-23) REMOVED")
        print(f"[YoloBackbone] Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass thủ công - xử lý skip connections
        Chỉ chạy layers 0 đến target_feature_layer_index
        """
        # Cache outputs của các layers (cho skip connections)
        outputs = {}
        
        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            from_idx = self.layer_from_info[idx]
            
            # Xác định input cho layer này
            if idx == 0:
                # Layer đầu tiên nhận input gốc
                layer_input = x
            elif isinstance(from_idx, list):
                # Skip connection - Concat từ nhiều layers
                inputs_to_concat = []
                for f in from_idx:
                    if f == -1:
                        # -1 nghĩa là layer ngay trước đó
                        inputs_to_concat.append(outputs[idx - 1])
                    elif f < 0:
                        # Negative index (relative)
                        inputs_to_concat.append(outputs[idx + f])
                    else:
                        # Absolute index
                        inputs_to_concat.append(outputs[f])
                layer_input = inputs_to_concat
            elif from_idx == -1:
                # Layer trước đó
                layer_input = outputs[idx - 1]
            else:
                # Absolute index
                layer_input = outputs[from_idx]
            
            # Forward qua layer
            outputs[idx] = layer(layer_input)
        
        # Lấy output của target layer
        out = outputs[self.target_feature_layer_index]
        
        return out if out.dim() == 4 else out.unsqueeze(0)
    
    def train(self, mode=True):
        super().train(mode)
        return self
    
    def eval(self):
        super().eval()
        return self
    
import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Model_YOLO_RVT.utils import DEVICE

def create_old_backbone(model_path, target_layer=13):
    """Tạo backbone cũ (hook + early exit) để so sánh"""
    from ultralytics import YOLO
    import torch.nn as nn
    
    class EarlyExitException(Exception):
        pass
    
    class YoloBackboneOld(nn.Module):
        def __init__(self, model_path, target_feature_layer_index=13):
            super().__init__()
            _temp_yolo = YOLO(model_path)
            self.yolo_detection_model = _temp_yolo.model
            self.target_feature_layer_index = target_feature_layer_index
            self._fmap_out = None
            self._use_early_exit = True
            
            layer = self.yolo_detection_model.model[target_feature_layer_index]
            self._hook_handle = layer.register_forward_hook(self._hook_fn)
            self.to(DEVICE)
        
        def _hook_fn(self, module, input_val, output_val):
            if isinstance(output_val, torch.Tensor):
                self._fmap_out = output_val
            if self._use_early_exit:
                raise EarlyExitException()
        
        def forward(self, x):
            self._fmap_out = None
            try:
                _ = self.yolo_detection_model(x)
            except EarlyExitException:
                pass
            return self._fmap_out
        
        def eval(self):
            super().eval()
            self._use_early_exit = True
            return self
    
    return YoloBackboneOld(model_path, target_layer)


def create_new_backbone(model_path, target_layer=13):
    """Tạo backbone mới (optimized) để so sánh"""
    from ultralytics import YOLO
    import torch.nn as nn
    
    class YoloBackboneNew(nn.Module):
        def __init__(self, model_path, target_feature_layer_index=13):
            super().__init__()
            _temp_yolo = YOLO(model_path)
            full_sequential = _temp_yolo.model.model
            
            self.target_feature_layer_index = target_feature_layer_index
            self.layers = nn.ModuleList()
            self.layer_from_info = []
            
            for i in range(target_feature_layer_index + 1):
                self.layers.append(full_sequential[i])
                self.layer_from_info.append(getattr(full_sequential[i], 'f', -1))
            
            del _temp_yolo, full_sequential
            self.to(DEVICE)
        
        def forward(self, x):
            outputs = {}
            
            for idx in range(len(self.layers)):
                layer = self.layers[idx]
                from_idx = self.layer_from_info[idx]
                
                if idx == 0:
                    layer_input = x
                elif isinstance(from_idx, list):
                    inputs_to_concat = []
                    for f in from_idx:
                        if f == -1:
                            inputs_to_concat.append(outputs[idx - 1])
                        elif f < 0:
                            inputs_to_concat.append(outputs[idx + f])
                        else:
                            inputs_to_concat.append(outputs[f])
                    layer_input = inputs_to_concat
                elif from_idx == -1:
                    layer_input = outputs[idx - 1]
                else:
                    layer_input = outputs[from_idx]
                
                outputs[idx] = layer(layer_input)
            
            return outputs[self.target_feature_layer_index]
    
    return YoloBackboneNew(model_path, target_layer)


def benchmark(backbone, name, x, num_runs=30):
    """Benchmark một backbone"""
    backbone.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = backbone(x)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        
        t1 = time.perf_counter()
        with torch.no_grad():
            out = backbone(x)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        
        t2 = time.perf_counter()
        times.append((t2 - t1) * 1000)
    
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"\n[{name}]")
    print(f"  Output shape: {out.shape}")
    print(f"  Avg time: {avg:.2f}ms ± {std:.2f}ms")
    print(f"  Min time: {min_t:.2f}ms")
    print(f"  Max time: {max_t:.2f}ms")
    
    return avg, out


def test_output_consistency(out1, out2):
    """Kiểm tra output có giống nhau không"""
    if out1.shape != out2.shape:
        print(f"\n[OUTPUT CONSISTENCY]")
        print(f"  ❌ Shape mismatch: {out1.shape} vs {out2.shape}")
        return False
    
    diff = torch.abs(out1 - out2).max().item()
    mean_diff = torch.abs(out1 - out2).mean().item()
    
    print(f"\n[OUTPUT CONSISTENCY]")
    print(f"  Max difference: {diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Match: {'✅ YES (identical)' if diff < 1e-5 else '⚠️ CLOSE (numerical precision)' if diff < 1e-3 else '❌ NO'}")
    
    return diff < 1e-3


def count_params(model):
    """Đếm số parameters"""
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 70)
    print("BENCHMARK: YOLO BACKBONE OPTIMIZATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    MODEL_PATH = 'Model_YOLO_RVT/yolov11s-pytorch-default-v1/best.pt'
    INPUT_SIZE = (480, 480)
    
    # Dummy input
    x = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(DEVICE)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input size: {INPUT_SIZE}")
    
    # Load OLD backbone (hook + early exit)
    print("\n" + "-" * 70)
    print("Loading OLD backbone (hook + EarlyExit)...")
    backbone_old = create_old_backbone(MODEL_PATH, target_layer=13)
    backbone_old.to(DEVICE)
    print(f"  Parameters: {count_params(backbone_old):,}")
    
    # Load NEW backbone (optimized - no hook)
    print("\n" + "-" * 70)
    print("Loading NEW backbone (optimized - no hook, no detection head)...")
    backbone_new = create_new_backbone(MODEL_PATH, target_layer=13)
    backbone_new.to(DEVICE)
    print(f"  Parameters: {count_params(backbone_new):,}")
    
    # Benchmark OLD
    print("\n" + "-" * 70)
    avg_old, out_old = benchmark(backbone_old, "OLD (Hook + EarlyExit)", x)
    
    # Benchmark NEW
    print("\n" + "-" * 70)
    avg_new, out_new = benchmark(backbone_new, "NEW (Optimized - No Hook)", x)
    
    # So sánh output
    print("\n" + "-" * 70)
    test_output_consistency(out_old, out_new)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    param_reduction = (count_params(backbone_old) - count_params(backbone_new)) / count_params(backbone_old) * 100
    speedup = avg_old / avg_new
    
    print(f"  Parameters:")
    print(f"    OLD: {count_params(backbone_old):,}")
    print(f"    NEW: {count_params(backbone_new):,}")
    print(f"    Reduction: {param_reduction:.1f}%")
    print(f"\n  Speed:")
    print(f"    OLD: {avg_old:.2f}ms")
    print(f"    NEW: {avg_new:.2f}ms")
    print(f"    Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    
    if speedup > 1.2:
        print(f"\n  ✅ NEW backbone NHANH HƠN {(speedup-1)*100:.1f}%!")
    elif speedup > 0.95:
        print(f"\n  ⚠️ Tốc độ tương đương (cần test trên GPU mạnh hơn)")
    else:
        print(f"\n  ❌ NEW backbone chậm hơn - cần kiểm tra lại")
    
    print("\n" + "=" * 70)
    print("Ghi chú: Speedup sẽ rõ ràng hơn trên GPU mạnh (RTX 3050 Ti)")
    print("vì giảm Python overhead và loại bỏ Detection Head")
    print("=" * 70)


if __name__ == "__main__":
    main()