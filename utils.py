def load_data():
    with open('rt-polaritydata/rt-polarity.pos', 'r', encoding='latin-1') as f:
        pos_lines = f.readlines()

    with open('rt-polaritydata/rt-polarity.neg', 'r', encoding='latin-1') as f:
        neg_lines = f.readlines()

    pos_labels = [1] * len(pos_lines)
    neg_labels = [0] * len(neg_lines)

    train_texts = pos_lines[:4000] + neg_lines[:4000]
    val_texts = pos_lines[4000:4500] + neg_lines[4000:4500]
    test_texts = pos_lines[4500:] + neg_lines[4500:]

    train_labels = pos_labels[:4000] + neg_labels[:4000]
    val_labels = pos_labels[4000:4500] + neg_labels[4000:4500]
    test_labels = pos_labels[4500:] + neg_labels[4500:]

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
