def main():
    label_file = "dataset/Syn90k/test_label.txt"
    pred_file = "../PaddleOCR/paddle_rec.txt"
    with open(label_file) as f:
        label = f.read().strip().split("\n")
    labels = [i.split() for i in label]
    with open(pred_file) as f:
        preds = f.read().strip().split("\n")
    count = 0
    for label, pred in zip(labels, preds):
        if label[1][1:].islower() and pred.isupper() and len(pred) >= 3:
            print(label, pred)
            count += 1
    print(count, len(preds))
    pass


if __name__ == '__main__':
    main()
