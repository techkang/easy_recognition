import torch
import torch.nn as nn

from model.backbone.transformer import PositionalEncoding


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p), nn.BatchNorm2d(out_c), nn.ReLU(True))


def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode == 'nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


class PositionAttnDecoder(nn.Module):
    def __init__(self, cfg, in_channels=512, num_channels=64,
                 mode='nearest', num_classes=36):
        super().__init__()
        self.max_length = cfg.charset.target_length
        self.k_encoder = nn.Sequential(
            # encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(in_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, scale_factor=2, mode=mode),
            # decoder_layer(num_channels, in_channels, scale_factor=(1, 2), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=self.max_length * 2)
        self.project = nn.Linear(in_channels, in_channels)

        self.cls = nn.Linear(in_channels, num_classes)

    def forward(self, feat, out_enc, targets_dict, image_meta):
        if out_enc is not None:
            x = out_enc
        else:
            x = feat
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)
        feature = attn_vecs
        pred = self.cls(attn_vecs)
        pred = pred.permute(0, 2, 1)

        return {"feature": feature, "pred": pred}
