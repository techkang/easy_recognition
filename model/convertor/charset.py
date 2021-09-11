import string

from transformers import BertTokenizer


class Charset:
    def __init__(self, corpus, cfg):
        if cfg.charset.use_space and " " not in corpus:
            corpus += " "
        self.blank = 0
        self.unknown = 1
        self.start_of_sequence = 2
        self.end_of_sequence = 3
        self.corpus = dict(zip(corpus, range(4, len(corpus) + 4)))
        self.single_char = [i for i in self.corpus if len(i) == 1]
        self.index = dict(zip(range(4, len(corpus) + 4), corpus))
        self.index[0] = ''
        self.index[1] = '😨'
        self.index[2] = ''
        self.index[3] = ''
        self.num_classes = len(self.index)
        self.target_length = cfg.charset.target_length

    def __getitem__(self, item):
        return self.index[item]

    def to_str(self, indexes):
        return ''.join(self.index[int(i)] if int(i) < self.num_classes else self.index[1] for i in indexes)

    def to_label(self, s):
        origin = [self.corpus.get(c, 1) for c in s]
        if self.target_length:
            origin += [0] * max(0, self.target_length - len(origin))
        return origin

    def filter_unknown(self, strings):
        if isinstance(strings, str):
            return "".join([i if i in self.corpus else self.index[1] for i in strings])
        else:
            return ["".join([i if i in self.corpus else self.index[1] for i in s]) for s in strings]

    def __len__(self):
        return len(self.index)


class BertCharset:
    def __init__(self, cfg):
        self.tokenizer = BertTokenizer.from_pretrained('output/huggingface/bert-base-chinese', do_lower_case=False)

        self.target_length = cfg.charset.target_length
        self.num_classes = len(self.tokenizer.vocab)
        self.blank = self.tokenizer.pad_token_id
        self.unk_token = '😨'
        self.unk_token_id = self.tokenizer.unk_token_id

        self.corpus = self.tokenizer.vocab
        self.single_char = [i for i in self.corpus if len(i) == 1]
        self.index = {v: k for k, v in self.tokenizer.vocab.items()}
        self.index[self.unk_token_id] = self.unk_token
        for char in (self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token):
            self.index[self.corpus[char]] = ""

    def __len__(self):
        return len(self.tokenizer)

    def to_str(self, indexes):
        res = [self.index.get(int(i), self.unk_token) for i in indexes]
        return "".join(res)

    def to_label(self, s):
        origin = [self.corpus.get(c, self.unk_token_id) for c in s]
        origin = [self.tokenizer.cls_token_id, *origin, self.tokenizer.sep_token_id]
        if self.target_length:
            origin += [0] * max(0, self.target_length - len(origin))
        return origin

    def filter_unknown(self, strings):
        if isinstance(strings, str):
            res = "".join([i if i in self.tokenizer.vocab else self.unk_token for i in strings])
        else:
            res = [self.filter_unknown(s) for s in strings]
        return res


class LowerCharset(Charset):
    def __init__(self, cfg):
        corpus = string.ascii_lowercase + string.digits
        super().__init__(corpus, cfg)

    def filter_unknown(self, strings):
        if isinstance(strings, str):
            strings = strings.lower()
            return "".join([i if i in self.corpus else self.index[1] for i in strings])
        else:
            strings = [i.lower() for i in strings]
            return ["".join([i if i in self.corpus else self.index[1] for i in s]) for s in strings]


class AsciiCharset(Charset):
    def __init__(self, cfg):
        corpus = string.ascii_letters + string.punctuation + string.digits
        super().__init__(corpus, cfg)


class FileCharset(Charset):
    def __init__(self, cfg):
        with open(cfg.charset.corpus, 'r', encoding='utf8') as f:
            corpus = []
            for line in f:
                if line:
                    if len(line.strip().split()) == 1:
                        corpus.append(" ")
                    else:
                        corpus.append(line.strip().split()[1])
        super().__init__(corpus, cfg)
