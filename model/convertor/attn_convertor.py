import torch as t


class AttnConvertor:

    def __init__(self, charset):
        self.charset = charset

    def str2tensor(self, strings):
        """
        Convert text-strings into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))
        """

        tensors = []
        attn_tensors = []
        for string in strings:
            index = self.charset.to_label(string)
            src_target = t.LongTensor(index)
            tensors.append(src_target)
            attn_tensor = t.roll(t.LongTensor(index), 1)
            attn_tensor[0] = self.charset.start_of_sequence
            attn_tensors.append(attn_tensor)
        tensors = t.stack(tensors)
        attn_tensors = t.stack(attn_tensors)
        target_lengths = t.IntTensor([len(s) + 1 for s in strings])
        return {'targets': tensors, "attn_targets": attn_tensors, "target_lengths": target_lengths}

    def tensor2str(self, outputs, img_metas=None):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        if outputs.shape[2] == self.charset.num_classes:
            pred = outputs.argmax(2)
        else:
            pred = outputs.argmax(1)
        str_converted = [self.charset.to_str(index) for index in pred]
        return str_converted

class BertConvertor(AttnConvertor):
    def str2tensor(self, strings):

        tensors = []
        for string in strings:
            index = self.charset.to_label(string)
            index = t.LongTensor(index)
            tensors.append(index)
        tensors = t.stack(tensors)
        target_lengths = t.IntTensor([len(s) + 2 for s in strings])
        return {'targets': tensors, "target_lengths": target_lengths}

