"""
═══════════════════════════════════════════════════════════════════════
  TOXIC COMMENT CLASSIFIER — Built from Scratch with Self-Attention
  Problem: Given a sentence, classify it as TOXIC or SAFE
  Architecture: Embedding → Self-Attention → Mean Pool → Classifier
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import re
from collections import Counter

# ─────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────

RAW_DATA = [
    # (sentence, label)  — 1 = toxic, 0 = safe
    ("you are an idiot and nobody likes you", 1),
    ("i hate everything about this place", 1),
    ("go away you stupid fool", 1),
    ("this is absolutely terrible and disgusting", 1),
    ("i want to destroy you completely", 1),
    ("shut up you are worthless", 1),
    ("everyone despises people like you", 1),
    ("you disgust me deeply", 1),
    ("what a pathetic loser you are", 1),
    ("you should be ashamed of yourself", 1),
    ("i despise your attitude entirely", 1),
    ("nobody wants you around here", 1),
    ("the weather is lovely today", 0),
    ("i really enjoyed reading that book", 0),
    ("thank you so much for your help", 0),
    ("the food at that restaurant was delicious", 0),
    ("she is a very talented artist", 0),
    ("i love spending time with my family", 0),
    ("congratulations on your achievement today", 0),
    ("this movie made me feel happy", 0),
    ("you did a great job on the project", 0),
    ("the sunset was absolutely beautiful tonight", 0),
    ("i appreciate your kindness and support", 0),
    ("learning new things is always exciting", 0),
]

# ─────────────────────────────────────────────────────────────────────
# 2. TOKENIZER & VOCABULARY
# ─────────────────────────────────────────────────────────────────────

class Tokenizer:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build(self, sentences):
        counts = Counter(w for s in sentences for w in self._tok(s))
        vocab = [self.PAD, self.UNK] + [w for w, c in counts.items() if c >= self.min_freq]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def _tok(self, s):
        return re.sub(r"[^a-z ]", "", s.lower()).split()

    def encode(self, s, max_len=12):
        ids = [self.word2idx.get(w, 1) for w in self._tok(s)]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────
# 3. MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def binary_cross_entropy(y_hat, y):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


# ─────────────────────────────────────────────────────────────────────
# 4. SELF-ATTENTION LAYER
# ─────────────────────────────────────────────────────────────────────

class SelfAttentionLayer:
    """
    Scaled Dot-Product Self-Attention
      Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
      Attn = softmax(Q @ Kᵀ / √d_k) @ V
    """
    def __init__(self, embed_dim, d_k, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (embed_dim + d_k))
        self.Wq = rng.normal(0, scale, (embed_dim, d_k))
        self.Wk = rng.normal(0, scale, (embed_dim, d_k))
        self.Wv = rng.normal(0, scale, (embed_dim, d_k))
        self.d_k = d_k
        # Cache for backprop
        self._cache = {}

    def forward(self, X):
        """X: (seq_len, embed_dim) → output: (seq_len, d_k)"""
        Q = X @ self.Wq                            # (seq, d_k)
        K = X @ self.Wk
        V = X @ self.Wv
        scores = Q @ K.T / np.sqrt(self.d_k)       # (seq, seq)
        A = softmax(scores)                         # attention weights
        out = A @ V                                 # (seq, d_k)
        self._cache = dict(X=X, Q=Q, K=K, V=V, A=A, scores=scores)
        return out, A

    def backward(self, dout, lr):
        """Simple gradient step for Wq, Wk, Wv."""
        c = self._cache
        X, Q, K, V, A = c['X'], c['Q'], c['K'], c['V'], c['A']

        # Gradient w.r.t. V
        dV = A.T @ dout
        dA = dout @ V.T

        # Gradient through softmax
        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores /= np.sqrt(self.d_k)

        # Gradients w.r.t. Q, K
        dQ = dscores @ K
        dK = dscores.T @ Q

        # Gradients w.r.t. weight matrices
        dWq = X.T @ dQ
        dWk = X.T @ dK
        dWv = X.T @ dV

        # Gradient clipping
        for dW in [dWq, dWk, dWv]:
            np.clip(dW, -1.0, 1.0, out=dW)

        self.Wq -= lr * dWq
        self.Wk -= lr * dWk
        self.Wv -= lr * dWv

        # Gradient back to X
        dX = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dX


# ─────────────────────────────────────────────────────────────────────
# 5. FULL MODEL
# ─────────────────────────────────────────────────────────────────────

class ToxicClassifier:
    """
    Architecture:
      Token IDs → Embedding → Self-Attention → Mean Pool → Linear → Sigmoid → P(toxic)
    """
    def __init__(self, vocab_size, embed_dim=16, d_k=8, seed=0):
        rng = np.random.default_rng(seed)
        # Embedding table
        self.E = rng.normal(0, 0.1, (vocab_size, embed_dim))
        # Self-Attention
        self.attn = SelfAttentionLayer(embed_dim, d_k, seed=seed)
        # Classifier head: d_k → 1
        self.W_cls = rng.normal(0, 0.1, (d_k, 1))
        self.b_cls = np.zeros(1)
        self._cache = {}

    def forward(self, token_ids):
        """
        token_ids: list[int] of length seq_len
        Returns scalar probability P(toxic)
        """
        X = self.E[token_ids]                       # (seq, embed_dim)
        attn_out, A = self.attn.forward(X)           # (seq, d_k)
        pooled = attn_out.mean(axis=0, keepdims=True) # (1, d_k)  — mean pool
        logit = pooled @ self.W_cls + self.b_cls      # (1, 1)
        prob = sigmoid(logit).item()
        self._cache = dict(token_ids=token_ids, X=X,
                           attn_out=attn_out, pooled=pooled, logit=logit)
        return prob, A

    def backward(self, prob, label, lr):
        c = self._cache
        # Loss gradient: d(BCE)/d(logit)
        dlogit = np.array([[prob - label]])            # (1,1)

        # Head gradients
        pooled = c['pooled']
        dW_cls = pooled.T @ dlogit
        db_cls = dlogit.sum(axis=0)
        dpooled = dlogit @ self.W_cls.T               # (1, d_k)

        np.clip(dW_cls, -1, 1, out=dW_cls)
        self.W_cls -= lr * dW_cls
        self.b_cls -= lr * db_cls

        # Back through mean pool
        seq_len = c['attn_out'].shape[0]
        dattn_out = np.repeat(dpooled, seq_len, axis=0) / seq_len  # (seq, d_k)

        # Back through Self-Attention
        dX = self.attn.backward(dattn_out, lr)

        # Back through embedding (sparse update)
        for i, idx in enumerate(c['token_ids']):
            grad = np.clip(dX[i], -1, 1)
            self.E[idx] -= lr * grad


# ─────────────────────────────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────────────────────────────

def train(epochs=300, lr=0.03, max_len=12):
    sentences, labels = zip(*RAW_DATA)

    tok = Tokenizer().build(sentences)
    encoded = [tok.encode(s, max_len) for s in sentences]
    model = ToxicClassifier(vocab_size=len(tok), embed_dim=16, d_k=8)

    print("═" * 58)
    print("  TOXIC COMMENT CLASSIFIER — Training")
    print(f"  Vocab: {len(tok)} words | Samples: {len(sentences)}")
    print("═" * 58)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        # Shuffle
        order = np.random.permutation(len(encoded))
        for i in order:
            prob, _ = model.forward(encoded[i])
            loss = binary_cross_entropy(np.array([prob]), np.array([labels[i]]))
            total_loss += loss
            model.backward(prob, labels[i], lr)
            correct += int((prob >= 0.5) == labels[i])

        acc = correct / len(encoded) * 100
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(encoded):.4f} | Acc: {acc:.1f}%")

    print("═" * 58)
    print("  Training complete!\n")
    return model, tok


# ─────────────────────────────────────────────────────────────────────
# 7. EVALUATE & PREDICT
# ─────────────────────────────────────────────────────────────────────

def predict(model, tok, sentence, max_len=12):
    ids = tok.encode(sentence, max_len)
    prob, attn_weights = model.forward(ids)
    label = "🔴 TOXIC" if prob >= 0.5 else "🟢 SAFE"
    confidence = prob if prob >= 0.5 else 1 - prob
    return prob, label, confidence, attn_weights

def evaluate(model, tok):
    print("  EVALUATION ON TRAINING SET")
    print("─" * 58)
    sentences, labels = zip(*RAW_DATA)
    correct = 0
    for s, l in zip(sentences, labels):
        prob, label, conf, _ = predict(model, tok, s)
        status = "✓" if (prob >= 0.5) == l else "✗"
        print(f"  {status} [{label}] {conf*100:.0f}% — \"{s[:45]}\"")
        correct += int((prob >= 0.5) == l)
    print("─" * 58)
    print(f"  Final Accuracy: {correct}/{len(sentences)} = {correct/len(sentences)*100:.1f}%\n")

def demo(model, tok):
    test_sentences = [
        "you are such a terrible person",
        "i really appreciate your effort today",
        "this is the worst thing i have ever seen",
        "the team did an amazing job this week",
        "go away nobody wants you here",
        "happy birthday have a wonderful day",
    ]
    print("  DEMO — New Sentences")
    print("─" * 58)
    for s in test_sentences:
        prob, label, conf, attn = predict(model, tok, s)
        print(f"  {label} ({conf*100:.0f}%) — \"{s}\"")
    print("─" * 58)

def interactive(model, tok):
    print("\n  INTERACTIVE MODE — type a sentence, press Enter.")
    print("  (type 'quit' to exit)\n")
    while True:
        try:
            s = input("  > ").strip()
            if s.lower() in ("quit", "exit", "q"):
                break
            if not s:
                continue
            prob, label, conf, _ = predict(model, tok, s)
            print(f"    → {label}  |  Confidence: {conf*100:.1f}%  |  Raw prob: {prob:.4f}\n")
        except (KeyboardInterrupt, EOFError):
            break


# ─────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(99)
    model, tok = train(epochs=300, lr=0.03)
    evaluate(model, tok)
    demo(model, tok)
    interactive(model, tok)
