import torch.nn as nn

from tools.init import xavier_init


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super().__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNNDecoder(nn.Module):

    def __init__(self, cfg, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        in_channels = cfg.crnn.inner_channel
        self.dropout = nn.Dropout(0.1)
        self.rnn_flag = True

        if self.rnn_flag:
            self.decoder = nn.Sequential(
                BidirectionalLSTM(in_channels, 256, 256),
                BidirectionalLSTM(256, 256, in_channels))
        else:
            self.decoder = nn.Conv2d(
                in_channels, in_channels, kernel_size=1, stride=1)
        self.fc = nn.Linear(in_channels, self.num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward(self, feat, out_enc, targets_dict, img_metas):
        assert feat.size(2) == 1, 'feature height must be 1'
        # B*L*1*C
        feat = self.dropout(feat)
        if self.rnn_flag:
            x = feat.squeeze(2)  # [N, C, W]
            feat = x.permute(2, 0, 1)  # [W, N, C]
            feat = self.decoder(feat)  # [W, N, C]
            feat = feat.permute(1, 0, 2)
        else:
            feat = self.decoder(feat)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            n, w, c, h = feat.size()
            feat = feat.reshape(n, w, c * h)
        outputs = self.fc(feat)
        return {"feature": feat, "pred": outputs}
