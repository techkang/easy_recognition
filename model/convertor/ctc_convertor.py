import numpy as np
import torch

from model.convertor.charset import Charset


class CTCConvertor:

    def __init__(self, charset: Charset):
        self.charset = charset

    def str2tensor(self, strings):
        """Convert text-strings to ctc-loss input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            dict (str: tensor | list[tensor]):
                tensors (list[tensor]): [torch.Tensor([1,2,3,3,4]),
                    torch.Tensor([5,4,6,3,7])].
                flatten_targets (tensor): torch.Tensor([1,2,3,3,4,5,4,6,3,7]).
                target_lengths (tensor): torch.IntTensot([5,5]).
        """

        tensors = []
        for string in strings:
            index = self.charset.to_label(string)
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
        tensors = torch.stack(tensors)
        target_lengths = torch.IntTensor([len(s) for s in strings])

        return {
            'targets': tensors,
            'target_lengths': target_lengths
        }

    def tensor2str(self, pred):
        if pred.ndim == 3:
            pred = pred.argmax(2)
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        output = np.zeros_like(pred, dtype=np.int) + self.charset.blank
        for i in range(pred.shape[0]):
            valid = 0
            previous = self.charset.blank
            for j in range(pred.shape[1]):
                c = pred[i][j]
                if c != previous and c != self.charset.blank:
                    output[i][valid] = c
                    valid += 1
                previous = c
        str_converted = [self.charset.to_str(index) for index in output]
        return str_converted
