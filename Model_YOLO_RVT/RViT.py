import torch
import torch.nn as nn
import random
from torch.nn import MultiheadAttention
import time
from utils import  NUM_CLASSES, DEVICE, SOS_TOKEN, EOS_TOKEN, MAX_SEQ_LENGTH

#--- Simplified RViT ---
class RViT(nn.Module):
    def __init__(self, yolo_channels=256, d_model=512, num_patches=1600, n_heads=8, num_encoder_layers=3, dim_feedforward=2048, dropout_rate=0.3):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Conv2d(yolo_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate if dropout_rate > 0 else 0)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.region_q = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.embed = nn.Embedding(NUM_CLASSES, d_model)
        self.gru_num_layers = 1
        self.gru = nn.GRU(d_model, d_model, num_layers=self.gru_num_layers, batch_first=True,
                          dropout=dropout_rate if self.gru_num_layers > 1 else 0)
        self.attn = MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate if dropout_rate > 0 else 0),
            nn.Linear(2 * d_model, NUM_CLASSES)
        )

    def forward(self, fmap, target=None, teach_ratio=0.5, forced_output_length=10):
        t1 = time.perf_counter() # Bắt đầu đo thời gian RViT
        b = fmap.size(0)
        x = self.proj(fmap)
        x = x.flatten(2).permute(0, 2, 1)

        current_num_patches = x.size(1)
        expected_pos_embed_len = current_num_patches + 1
        
        if self.pos_embed.size(1) != expected_pos_embed_len:
            if self.pos_embed.size(1) > expected_pos_embed_len:
                pos_embed_to_add = self.pos_embed[:, :expected_pos_embed_len, :]
            else:
                raise ValueError(f"RViT pos_embed second dim {self.pos_embed.size(1)} is smaller than required {expected_pos_embed_len}")
        else:
            pos_embed_to_add = self.pos_embed

        q = self.region_q.expand(b, -1, -1)
        x = torch.cat([q, x], dim=1)
        x = x + pos_embed_to_add

        enc = self.encoder(x)
        region_feat, spatial_feats = enc[:, 0], enc[:, 1:]
        
        if forced_output_length is not None:
            max_gen_len = forced_output_length
        elif target is not None:
            max_gen_len = target.size(1) - 1
        else:
            max_gen_len = MAX_SEQ_LENGTH - 1

        h = region_feat.unsqueeze(0).contiguous()
        current_input_tokens = torch.full((b,), SOS_TOKEN, device=DEVICE, dtype=torch.long)
        outputs_logits = []

        finished_sequences_tracker = None
        if target is None and forced_output_length is None:
            finished_sequences_tracker = torch.zeros(b, dtype=torch.bool, device=DEVICE)
        
        t2 = time.perf_counter() # Kết thúc đo thời gian chuẩn bị RViT
        print(f"RViT preparation time: {(t2 - t1)*1000:.2f} ms")
        for t in range(max_gen_len):
            emb = self.embed(current_input_tokens).unsqueeze(1)
            g, h = self.gru(emb, h)
            a, _ = self.attn(g, spatial_feats, spatial_feats)
            comb = torch.cat([g.squeeze(1), a.squeeze(1)], dim=-1)
            logits_step = self.fc(comb)
            outputs_logits.append(logits_step)

            if target is not None and random.random() < teach_ratio:
                next_input_candidate = target[:, t + 1]
            else:
                next_input_candidate = logits_step.argmax(-1)

            if finished_sequences_tracker is not None:
                eos_predicted_this_step = (next_input_candidate == EOS_TOKEN)
                finished_sequences_tracker |= eos_predicted_this_step
                current_input_tokens = torch.where(finished_sequences_tracker,
                                                 torch.tensor(EOS_TOKEN, device=DEVICE, dtype=torch.long),
                                                 next_input_candidate)
                if finished_sequences_tracker.all():
                    break
            else:
                current_input_tokens = next_input_candidate
        
        output = torch.stack(outputs_logits, dim=1)
        t3 = time.perf_counter() # Kết thúc đo thời gian RViT
        print(f"RViT generation time: {(t3 - t2)*1000:.2f} ms")
        return output