import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """Implementation of loss module for CTC-loss based text recognition.

    Args:
        blank (int): Blank label. Default 0.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        zero_infinity (bool): Whether to zero infinite losses and
            the associated gradients. Default: False.
            Infinite losses mainly occur when the inputs
            are too short to be aligned to the targets.
    """

    def __init__(self, blank=0, reduction='mean', zero_infinity=True):
        super().__init__()
        assert isinstance(blank, int)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)

        self.blank = blank
        self.ctc_loss = nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, outputs, targets_dict):
        outputs = torch.log_softmax(outputs.float(), dim=2)
        bsz, seq_len = outputs.shape[:2]
        input_lengths = torch.full(size=(bsz,), fill_value=seq_len, dtype=torch.long)
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()  # T * N * C

        targets = targets_dict['targets']
        target_lengths = targets_dict['target_lengths']
        loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths, target_lengths)
        losses = dict(loss_ctc=loss_ctc)

        return losses


class CTCDropLoss(CTCLoss):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True):
        super().__init__(blank, "none", zero_infinity)
        self.zero_ratio = 0.1

    def forward(self, outputs, targets_dict):
        losses = super().forward(outputs, targets_dict)
        for k, v in losses.items():
            sorted_loss = v.sort(descending=True)[0]
            mask = torch.ones(len(sorted_loss), device=sorted_loss.device)
            mask[:int(len(sorted_loss)*self.zero_ratio)] = 0
            losses[k] = (mask * sorted_loss).mean()
        return losses
