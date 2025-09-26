import torch
import torch.nn as nn



# transformer_model = nn.Transformer(d_model=16, nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 16))
# tgt = torch.rand((20, 32, 16))
# out = transformer_model(src, tgt)
# print(out.shape)

print(torch.__version__)


encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 16)
out = transformer_encoder(src)
print(out)
