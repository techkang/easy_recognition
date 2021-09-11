import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder

from .transformer import PositionalEncoding, NoSATrmDecoderLayer


class BCNLanguage(nn.Module):
    def __init__(self, cfg, num_classes=36):
        super().__init__()
        d_model = 512
        nhead = 8
        d_inner = 2048
        dropout = 0.1
        activation = "relu"
        num_layers = 4
        self.d_model = d_model
        self.detach = True
        self.use_self_attn = False
        self.loss_weight = 1.
        self.max_length = cfg.charset.target_length  # additional stop token
        self.num_classes = num_classes

        self.proj = nn.Linear(num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length * 2)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length * 2)
        decoder_layer = NoSATrmDecoderLayer(d_model, nhead, d_inner, dropout, activation)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, num_classes)

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = t.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = t.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, batch):
        tokens, lengths = batch["targets"], batch["target_lengths"]
        if self.detach:
            tokens = tokens.detach()
        if tokens.ndim == 2:
            tokens = F.one_hot(tokens, num_classes=self.num_classes)*1.
        embed = self.proj(tokens)  # (N, T, E)
        seq_length = embed.shape[1]
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, seq_length)
        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(seq_length, tokens.device)

        output = self.model(qeury, embed, tgt_key_padding_mask=padding_mask, memory_mask=location_mask,
                            memory_key_padding_mask=padding_mask)  # (T, N, E)
        feature = output.permute(1, 0, 2)  # (N, T, E)

        pred = self.cls(feature)  # (N, T, C)
        pred = pred.permute(0, 2, 1)

        return feature, pred
