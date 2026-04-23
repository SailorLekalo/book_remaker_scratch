import random

from remake_my_book_from_scratch.tokenizer import indexate, load_and_split, build_dataset, encode_dataset
from tqdm import tqdm
import numpy as np

EMBEDDING_SIZE = 512
WINDOW_SIZE = 15  # например 3, 5, 7...

texts = load_and_split()
index = indexate(texts)
id2token = {v: k for k, v in index.items()}
embedding = np.random.randn(len(index), EMBEDDING_SIZE)

dataset = build_dataset(texts, window_size=WINDOW_SIZE)
dataset = encode_dataset(dataset, index)

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    a = np.sqrt(2 / np.pi)
    u = a * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    sech2 = 1 - t * t
    du_dx = a * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * sech2 * du_dx
class Layer:
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(out_size, in_size) * np.sqrt(2.0 / in_size) #подробнее изучить тему инициализации весов
        self.bias = np.zeros(out_size)

    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.bias
        #self.a = np.maximum(0, self.z)
        #self.a = np.where(self.z > 0, self.z, 0.01 * self.z)  # Leaky ReLu, from 90 epoch instead of ReLu
        self.a = gelu(self.z) #GeLu instead of LReLu after 200 epoch
        return self.a

    def backward(self, grad_out, lr=0.01, weight_decay = 0.1):
        dz = grad_out * gelu_grad(self.z)
        dW = np.outer(dz, self.x)
        dx = self.W.T @ dz
        self.W -= lr * (dW + weight_decay * self.W)

        db = dz
        self.bias -= lr * db

        return dx

    def save(self):
        return {
            "W": self.W,
            "bias": self.bias
        }

    def load(self, data):
        self.W = data["W"]
        self.bias = data["bias"]

class EmbeddingLayer:
    def __init__(self, embedding):
        self.embedding = embedding
        self.last_3 = None

    def forward(self, x):

        self.last_3 = x
        vecs = [self.embedding[i] for i in x]

        return np.concatenate(vecs, axis=0)

    def backward(self, grad, lr = 0.01):
        chunks = np.split(grad, len(self.last_3))

        for idx, g in zip(self.last_3, chunks):
            self.embedding[idx] -= lr * g

    def save(self):
        return self.embedding

    def load(self, emb):
        self.embedding = emb

class Model:
    def __init__(self, embedding, layers: list[Layer]):
        self.embedding_layer = EmbeddingLayer(embedding)
        self.layers = layers
        self.cache_logits = None

    def forward(self, a: list[int]):

        x = self.embedding_layer.forward(a)

        for layer in self.layers:
            x = layer.forward(x)

        self.cache_logits = x
        return softmax(x)

    def backward(self, grad, lr=0.01):

        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

        self.embedding_layer.backward(grad, lr)

    def save(self, path="model.npz"):
        data = {
            "embedding": self.embedding_layer.save(),
        }

        for i, layer in enumerate(self.layers):
            layer_data = layer.save()
            data[f"W_{i}"] = layer_data["W"]
            data[f"b_{i}"] = layer_data["bias"]

        np.savez(path, **data)

    def load(self, path="model.npz"):
        data = np.load(path, allow_pickle=True)

        self.embedding_layer.load(data["embedding"])
        for i, layer in enumerate(self.layers):
            layer.W = data[f"W_{i}"]
            layer.bias = data[f"b_{i}"]

model = Model(
    embedding,
    [
        Layer(EMBEDDING_SIZE*WINDOW_SIZE, 256),
        Layer(256, len(index)),
    ]
)

start_epoch = 312
epochs = 100000
lr = 0.0001
def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1 - np.dot(a, b)
def check(epoch, collect_errors=False):
    correct = 0
    total = 0
    errors = []

    for x, y in dataset:
        probs = model.forward(x)
        pred_token = np.argmax(probs)

        if pred_token == y:
            correct += 1
        else:
            if collect_errors:
                errors.append((x, y))

        total += 1

    accuracy = correct / total
    print(f"Epoch {epoch + 1}: test_accuracy = {accuracy:.4f} ({correct}/{total})")

    if collect_errors:
        return errors

    print(f"Epoch {epoch + 1}: test_accuracy = {accuracy:.4f} ({correct}/{total})")

model.load(f"checkpoint_{start_epoch}.npz")

def true_rank(probs, y):
    sorted_idx = np.argsort(probs)[::-1]  # от большего к меньшему
    rank = np.where(sorted_idx == y)[0][0]
    return rank

ranks = []

def train():
    for epoch in range(start_epoch+1, epochs):

        total_loss = 0


        error_dataset = check(epoch, collect_errors=True)
        error_dataset = error_dataset*100
        mixed_dataset = dataset
        random.shuffle(mixed_dataset)

        # tqdm показывает прогресс внутри эпохи
        pbar = tqdm(mixed_dataset, desc=f"Epoch {epoch + 1}/{epochs}")

        for x, y in pbar:

            # ---- forward ----
            pred = model.forward(x)

            # ---- loss (cross-entropy) ----
            loss = -np.log(pred[y] + 1e-9)

            if pred[y] < 0.1:
                loss *= 3.0
            elif pred[y] < 0.3:
                loss *= 2.0

            grad = pred.copy()
            grad[y] -= 1
            grad *= loss
            total_loss += loss
            # ---- backward ----
            model.backward(grad, lr)
            #print(np.max(pred), np.min(pred))
            # ---- live update tqdm ----
            pbar.set_postfix({
                "loss": float(loss),
                "avg_loss": float(total_loss) / (pbar.n + 1),
                "grad norm": np.linalg.norm(grad)
            })
        model.save(f"checkpoint_{epoch}.npz")
        print(f"Epoch {epoch + 1}: avg_loss = {total_loss / len(mixed_dataset)}")
        print("linalg norm of logits", np.linalg.norm(model.cache_logits))
        print("logits std:", np.std(model.cache_logits))
        print("logits max:", np.max(model.cache_logits))
        print("logits min:", np.min(model.cache_logits))

#train()
def generate(model, start_tokens, max_new_tokens=50, temperature=1.0):
    result = list(start_tokens)

    for _ in range(max_new_tokens):

        context = result[-WINDOW_SIZE:]

        if len(context) < WINDOW_SIZE:
            context = [0] * (WINDOW_SIZE - len(context)) + context

        pred = model.forward(context)

        result.append(np.argmax(pred))

    return result

def decode(tokens):
    return " ".join(id2token[t] for t in tokens)

def decode_context(context, id2token):
    return " ".join(id2token[i] for i in context if i in id2token)



start_text = "Глава 1. Тамара и Турнир Я вскинула руку в инстинктивном порыве заблокировать магию, которую не мог".lower().split()
start = [index[w] for w in start_text if w in index]

generated = generate(model, start_tokens=start, max_new_tokens=2277)

print(decode(generated))

