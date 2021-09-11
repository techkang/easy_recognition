import logging

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import model
import model.convertor.charset
from tools.comm import data_to
from tools.comm import freeze_model
from .base import CRNN as BaseVision
from .language_model import LanguageModel


class BaseLanguage(LanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.abi.freeze_language:
            freeze_model(self.bert_model)
            freeze_model(self.predictions)

    def forward(self, embed):
        feature = self.embeddings(embed)
        feature = self.bert_model(feature).last_hidden_state
        feature = self.predictions(feature)
        pred = self.cls(feature).permute(0, 2, 1)
        return {"feature": feature, "pred": pred}


class TFAlignment(nn.Module):

    def __init__(self, num_classes=36, d_model=512, nhead=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, num_classes)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, q_feature, k_v_feature):
        """
        :param q_feature:  (seq length, batch size, embed dim)
        :param k_v_feature: (seq length, batch size, embed dim)
        :return:
        """
        q_feature = q_feature.permute(1, 0, 2)
        k_v_feature = k_v_feature.permute(1, 0, 2)
        src2 = self.self_attn(q_feature, k_v_feature, k_v_feature)[0]
        src = q_feature + self.dropout1(src2)
        src = self.norm1(src)
        src = src.permute(1, 0, 2)
        pred = self.linear1(src)
        return pred


class BaseAlignment(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        d_model = 512

        self.w_att = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, language_feature, vision_feature):
        f = t.cat((language_feature, vision_feature), dim=2)
        f_att = t.sigmoid(self.w_att(f))
        output = f_att * vision_feature + (1 - f_att) * language_feature

        pred = self.cls(output)  # (N, T, C)
        return pred


class ABINetIterModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        charset = getattr(model.convertor.charset, cfg.charset.name)(cfg)

        self.iter_size = cfg.abi.iter_time
        self.max_length = cfg.charset.target_length

        self.vision = BaseVision(cfg)
        self.load_weights(self.vision, cfg.abi.vision_model_weights)

        self.language = BaseLanguage(cfg)
        self.load_weights(self.language, cfg.abi.language_model_weights)

        self.align_to_vision = True
        if cfg.abi.align == "cross":
            self.alignment = TFAlignment(num_classes=charset.num_classes)
        else:
            self.alignment = BaseAlignment(num_classes=charset.num_classes)
            self.align_to_vision = False

        self.ctc_loss = model.loss.CTCLoss(blank=charset.blank)
        self.ce_loss = model.loss.CELoss()

        self.ctc_convertor = model.convertor.CTCConvertor(charset)
        self.attn_convertor = model.convertor.AttnConvertor(charset)

        self.device = t.device(cfg.device)
        self.blank = charset.blank

    def load_weights(self, module: t.nn.Module, weight_path: str):
        if weight_path:
            state_dict = t.load(weight_path, map_location=t.device("cpu"))
            if 'model' in state_dict:
                state_dict = state_dict["model"]
            module.load_state_dict(state_dict, strict=True)
            logging.info(f"Successfully load state dict from {weight_path} for {module.__class__.__name__}")

    def ctc_rearrange(self, pred, feature):
        assert pred.ndim == 3
        if isinstance(pred, t.Tensor):
            pred = pred.detach()
        blank = t.zeros_like(feature)
        pred = pred.argmax(2)
        for i in range(pred.shape[0]):
            valid = 0
            previous = 0
            for j in range(pred.shape[1]):
                c = int(pred[i][j])
                if c != previous and c != 0:
                    blank[i][valid] = feature[i][j]
                    valid += 1
                previous = c
        return blank

    def forward(self, batch):
        vision_res = self.vision(batch)
        vision_feature = vision_res["feature"]
        vision_pred = vision_res["pred"]
        if isinstance(self.alignment, BaseAlignment):
            vision_feature = self.ctc_rearrange(vision_pred, vision_feature)
        align_pred = self.ctc_rearrange(vision_pred, vision_pred).detach()
        all_language_res, all_a_res = [], []
        for _ in range(self.iter_size):
            align_pred = align_pred.softmax(2)
            language_res = self.language(align_pred)
            all_language_res.append(language_res['pred'])
            if self.align_to_vision:
                align_pred = self.alignment(vision_feature, language_res['feature'])
            else:
                align_pred = self.alignment(language_res['feature'], vision_feature)
            all_a_res.append(align_pred)

        gt_labels = batch["text"]
        vision_target = data_to(self.ctc_convertor.str2tensor(gt_labels), self.device)
        language_target = data_to(self.attn_convertor.str2tensor(gt_labels), self.device)

        language_target["targets"] = F.pad(language_target["targets"], (0, all_language_res[0].shape[-1] -
                                                                        self.attn_convertor.charset.target_length))

        loss_dict = self.iter_loss(all_a_res, all_language_res, vision_pred, language_target, vision_target)

        if self.training:
            return loss_dict

        vision_pred_strings = self.ctc_convertor.tensor2str(vision_pred)
        language_pred_strings = self.attn_convertor.tensor2str(all_language_res[-1])
        if self.align_to_vision:
            pred_strings = self.ctc_convertor.tensor2str(align_pred)
        else:
            pred_strings = self.attn_convertor.tensor2str(align_pred)

        text = [vision_pred_strings, language_pred_strings, pred_strings, gt_labels]
        results = {"loss": loss_dict, "text": text, "dataset": batch["dataset"], "pred": pred_strings,
                   "label": gt_labels, "vision": vision_pred_strings, "language": language_pred_strings}
        return results

    def iter_loss(self, align_pred, language_pred, vision_pred, language_target, vision_target):
        loss_name = ("align", "language", "vision")
        pred_list = [align_pred, language_pred, [vision_pred]]
        target_list = [vision_target, language_target, vision_target]
        loss_list = [self.ctc_loss, self.ce_loss, self.ctc_loss]
        if not self.align_to_vision:
            target_list[0] = language_target
            loss_list[0] = self.ce_loss
        loss_dict = {}
        for name, pred, target, loss in zip(loss_name, pred_list, target_list, loss_list):
            total = sum(sum(loss(p, target).values()) for p in pred)
            total = total / len(pred)
            loss_dict[name + "_loss"] = total
        return loss_dict
