from collections import defaultdict

from .trainer import Trainer


class ABITrainer(Trainer):

    def collect_batch(self, pred_label_str_dict):
        acc = defaultdict(int)
        total = defaultdict(int)
        vision_acc = defaultdict(int)
        language_acc = defaultdict(int)
        for each in pred_label_str_dict:
            dataset, pred, vision, language, label = each["dataset"], each["pred"], each["vision"], each["language"], \
                                                     each["label"]
            for i in range(len(dataset)):
                acc[dataset[i]] += int(pred[i] == label[i])
                vision_acc[dataset[i]] += int(vision[i] == label[i])
                language_acc[dataset[i]] += int(language[i] == label[i])
                total[dataset[i]] += 1
        average_precision = sum(acc.values()) / sum(total.values())
        average_vision = sum(vision_acc.values()) / sum(total.values())
        average_language = sum(language_acc.values()) / sum(total.values())
        return {"score": average_precision,
                "score/vision": average_vision, "score/iter": average_precision, "score/language": average_language,
                **{"word/" + k: acc[k] / total[k] for k in acc},
                **{"vision/" + k: vision_acc[k] / total[k] for k in acc},
                **{"language/" + k: language_acc[k] / total[k] for k in acc},
                }
