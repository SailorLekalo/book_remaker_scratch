def load_and_split():
    with open("chapter_1.txt", "r", encoding="utf-8") as f:
        text = f.read().split()
        return [s.lower().strip() for s in text]

def indexate(texts):
    vocab = {}
    for w in texts:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab

def build_dataset(texts, window_size=3):
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size должен быть нечётным и >= 3")

    dataset = []

    for i in range(window_size, len(texts)):
        # окно любого размера: 3, 5, 7, ...
        x = texts[i - window_size:i]
        y = texts[i]

        dataset.append((x, y))

    return dataset
def encode_dataset(dataset, vocab):
    encoded = []
    for x, y in dataset:
        encoded.append((
            [vocab[w] for w in x],
            vocab[y]
        ))
    return encoded