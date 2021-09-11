import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertForMaskedLM, PretrainedConfig

import model
from model.backbone.transformer import PositionalEncoding
from tools.comm import data_to


class BCNLanguageModel(nn.Module):
    """Base class for encode-decode recognizer."""

    def __init__(self, cfg):
        super().__init__()

        charset = getattr(model.convertor.charset, cfg.charset.name)(cfg)

        self.preprocessor = None
        if cfg.model.preprocessor:
            self.preprocessor = getattr(model.preprocessor, cfg.model.preprocessor)(cfg)

        self.model = getattr(model.backbone, cfg.model.backbone)(cfg, num_classes=charset.num_classes)
        self.loss = getattr(model.loss, cfg.model.loss)(blank=charset.blank)
        self.label_convertor = getattr(model.convertor, cfg.model.convertor)(charset)
        self.device = t.device(cfg.device)

    def forward(self, batch):
        input_text, label_text = batch["input"], batch["label"]
        if self.preprocessor:
            input_text, label_text = self.preprocessor(input_text), self.preprocessor(label_text)

        input_text = self.label_convertor.charset.filter_unknown(input_text)
        label_text = self.label_convertor.charset.filter_unknown(label_text)

        input_dict = self.label_convertor.str2tensor(input_text)
        input_dict = data_to(input_dict, self.device)
        targets_dict = self.label_convertor.str2tensor(label_text)
        targets_dict = data_to(targets_dict, self.device)

        feature, pred = self.model(input_dict)

        loss_inputs = (pred, targets_dict)
        loss_dict = self.loss(*loss_inputs)
        if self.training:
            # loss_dict["feature"] = feature
            return loss_dict

        pred = pred.argmax(1)
        pred_strings = [self.label_convertor.charset.to_str(index) for index in pred]

        results = {"loss": loss_dict, "pred": pred_strings, "label": label_text, "dataset": batch["dataset"],
                   "feature": feature}
        return results


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        charset = getattr(model.convertor.charset, cfg.charset.name)(cfg)
        self.num_classes = charset.num_classes
        bert_model = AlbertForMaskedLM(
            PretrainedConfig.from_json_file("output/huggingface/albert_chinese_tiny/config.json"))
        in_dim = bert_model.albert.embeddings.word_embeddings.embedding_dim
        out_dim = bert_model.predictions.dense.in_features

        self.embeddings = nn.Sequential(
            nn.Linear(self.num_classes, in_dim),
            PositionalEncoding(in_dim, max_len=512),
            nn.LayerNorm(in_dim)
        )
        self.bert_model = bert_model.albert.encoder
        self.predictions = nn.Sequential(
            nn.Linear(out_dim, cfg.crnn.inner_channel),
            nn.LayerNorm(cfg.crnn.inner_channel))

        self.loss = getattr(model.loss, cfg.model.loss)(blank=charset.blank)
        self.label_convertor = getattr(model.convertor, cfg.model.convertor)(charset)

        self.cls = nn.Linear(cfg.crnn.inner_channel, charset.num_classes)
        self.device = t.device(cfg.device)

    def forward(self, batch):
        input_text, label_text = batch["input"], batch["label"]

        tokens = self.label_convertor.str2tensor(input_text)
        tokens = data_to(tokens, self.device)
        embed = F.one_hot(tokens["targets"], self.num_classes) * 1.

        feature = self.embeddings(embed)
        feature = self.bert_model(feature).last_hidden_state
        feature = self.predictions(feature)

        pred = self.cls(feature).permute(0, 2, 1)

        labels = self.label_convertor.str2tensor(label_text)
        labels = data_to(labels, self.device)
        loss_inputs = (pred, labels)
        loss_dict = self.loss(*loss_inputs)
        if self.training:
            # loss_dict["feature"] = feature
            return loss_dict

        pred_strings = self.label_convertor.tensor2str(pred)

        results = {"loss": loss_dict, "pred": pred_strings, "label": label_text, "dataset": batch["dataset"],
                   "feature": feature, "text": [input_text, pred_strings, label_text]}
        return results
