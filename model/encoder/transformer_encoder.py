import torch.nn as nn
from torch.nn import TransformerEncoderLayer

from model.backbone.transformer import PositionalEncoding


class TFEncoder(nn.Module):
    """Encode 2d feature map to 1d sequence."""

    def __init__(self,
                 cfg,
                 n_layers=6,
                 n_head=8,
                 d_model=512,
                 d_inner=256,
                 dropout=0.1,
                 ):
        super().__init__()
        self.d_model = d_model
        self.layer_stack = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_head, d_inner, dropout=dropout) for _ in range(n_layers)])
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8*32*8)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, img_metas=None):
        n, c, h, w = feat.size()
        feat = feat.view(n, c, h * w)

        output = feat.permute(2, 0, 1)  # sequence length, batch size, embed dim
        output = self.pos_encoder(output)
        for enc_layer in self.layer_stack:
            output = enc_layer(output)
        output = self.layer_norm(output)

        output = output.permute(1, 2, 0).contiguous()
        output = output.view(n, self.d_model, h, w)

        return output
