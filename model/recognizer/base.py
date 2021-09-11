import torch as t
import torch.nn as nn

import model
from tools.comm import data_to


class Recognizer(nn.Module):
    """Base class for encode-decode recognizer."""

    def __init__(self, cfg):
        super().__init__()

        charset = getattr(model.convertor.charset, cfg.charset.name)(cfg)
        # Preprocessor module, e.g., TPS
        self.preprocessor = None
        if cfg.model.preprocessor:
            self.preprocessor = getattr(model.preprocessor, cfg.model.preprocessor)(cfg)

        # Backbone
        self.backbone = getattr(model.backbone, cfg.model.backbone)(cfg)

        # Encoder module
        self.encoder = None
        if cfg.model.encoder:
            self.encoder = getattr(model.encoder, cfg.model.encoder)(cfg)

        # Decoder module
        self.decoder = getattr(model.decoder, cfg.model.decoder)(cfg, num_classes=charset.num_classes)

        # Loss
        self.loss = getattr(model.loss, cfg.model.loss)(blank=charset.blank)

        # converter
        self.label_convertor = getattr(model.convertor, cfg.model.convertor)(charset)

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        if self.preprocessor is not None:
            img = self.preprocessor(img)

        x = self.backbone(img)

        return x

    def forward(self, batch):
        image = batch["image"]
        image_meta = batch
        feat = self.extract_feat(image)

        gt_labels = image_meta["text"]
        gt_labels = self.label_convertor.charset.filter_unknown(gt_labels)
        targets_dict = self.label_convertor.str2tensor(gt_labels)
        targets_dict = data_to(targets_dict, feat.device)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, image_meta)

        out_dec = self.decoder(feat, out_enc, targets_dict, image_meta)
        if isinstance(out_dec, dict):
            out_dec = out_dec["pred"]

        loss_inputs = (
            out_dec,
            targets_dict,
        )
        losses = self.loss(*loss_inputs)
        if self.training:
            return losses

        pred_strings = self.label_convertor.tensor2str(data_to(out_dec, t.device("cpu")))

        results = {"loss": losses, "pred": pred_strings, "label": gt_labels, "dataset": image_meta["dataset"]}
        return results


class CRNN(Recognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self, cfg):
        super().__init__(cfg)
        charset = model.convertor.charset.FileCharset(cfg)
        self.backbone = model.backbone.ResT(cfg)
        self.decoder = model.decoder.CRNNDecoder(cfg, num_classes=charset.num_classes)

    def forward(self, batch):
        feat = self.extract_feat(batch["image"])
        return self.decoder(feat, None, None, batch)

