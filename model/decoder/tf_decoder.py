import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F


def get_pad_mask(seq, pad_idx):
    return seq == pad_idx


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    len_s = seq.size(1)
    subsequent_mask = t.triu(t.ones((len_s, len_s), device=seq.device, dtype=t.bool), diagonal=1)
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid=512, n_position=200):
        super().__init__()

        # Not a parameter
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = t.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = t.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = t.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = t.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        self.device = x.device
        return x + self.position_table[:, :x.size(1)].clone().detach()


class TFDecoder(nn.Module):
    """Transformer Decoder block with self attention mechanism."""

    def __init__(self, cfg, n_layers=6, d_embedding=512, n_head=8, d_model=512, d_inner=256, n_position=200,
                 dropout=0.1, num_classes=93, max_seq_len=40, start_idx=2, padding_idx=0, ):
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(num_classes, d_embedding, padding_idx=padding_idx)

        self.position_enc = PositionalEncoding(d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model, n_head, d_inner, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.classifier = nn.Linear(d_model, num_classes)

    def _attention(self, trg_seq, src):
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = get_subsequent_mask(trg_seq)
        tgt_key_padding_mask = get_pad_mask(trg_seq, pad_idx=self.padding_idx)
        output = tgt.permute(1, 0, 2)
        src = src.permute(1, 0, 2)
        for dec_layer in self.layer_stack:
            output = dec_layer(output, src, tgt_mask=trg_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.layer_norm(output)

        return output

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_test(*args, **kwargs)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        n, c, h, w = out_enc.size()
        out_enc = out_enc.view(n, c, h * w).permute(0, 2, 1)
        out_enc = out_enc.contiguous()
        targets = targets_dict['attn_targets'].to(out_enc.device)
        attn_output = self._attention(targets, out_enc)
        outputs = self.classifier(attn_output).permute(1, 2, 0)
        return outputs

    def forward_test(self, feat, out_enc, targets_dict, image_meta):
        n, c, h, w = out_enc.size()
        out_enc = out_enc.view(n, c, h * w).permute(0, 2, 1)
        out_enc = out_enc.contiguous()

        init_target_seq = t.full((n, self.max_seq_len + 1),
                                 self.padding_idx,
                                 device=out_enc.device,
                                 dtype=t.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self._attention(init_target_seq, out_enc)
            # bsz * seq_len * 512
            step_result = F.softmax(self.classifier(decoder_output[step]), dim=-1)
            # bsz * num_classes
            outputs.append(step_result)
            _, step_max_index = t.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = t.stack(outputs, dim=2)

        return outputs
