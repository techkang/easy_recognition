import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        targets = targets["targets"]
        if outputs.shape[2] != targets.shape[1] and outputs.shape[1] == targets.shape[1]:
            outputs = outputs.permute(0, 2, 1)
        ce_loss = self.ce_loss(outputs, targets)
        return {"ce_loss": ce_loss}
