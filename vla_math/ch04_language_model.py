"""演習 4: 言葉を数にする（トークン化と、いちばん小さい言語モデル）

実行: python3 vla_math/ch04_language_model.py
出力: 画面への表示だけ（図は作らない）
必要: numpy（GPU は不要. 数秒で終わる）

ロボットへの指示文（「赤いブロックを棚に置いて。」など）を、決まった型から 3000 文作って使う.
(1) 文字を 1 つずつトークンにする
(2) BPE（よく隣り合う 2 つを 1 つにまとめる操作の繰り返し）で、トークンを作る
(3) 1 つ前のトークンから次を当てる言語モデル（バイグラム）を、数え上げで学習し、
    見ていない文での交差エントロピーと perplexity を求める
(4) 学習した言語モデルから、温度を変えて文を作る
"""

from __future__ import annotations

from collections import Counter

import numpy as np

COLORS = ["赤い", "青い", "黄色い", "緑の", "白い"]
OBJECTS = ["ブロック", "コップ", "ボール", "箱", "ペン"]
PLACES = ["机の上", "棚", "かご", "トレイ", "左の箱"]
TEMPLATES = [
    "{c}{o}を{p}に置いて。",
    "{c}{o}を取って。",
    "{p}にある{c}{o}を{q}に移して。",
    "{c}{o}をつかんで{p}に入れて。",
]
BOS, EOS = "<始>", "<終>"
ALPHA = 0.001          # 一度も見ていない組にも、ごく小さい確率を残すための下駄


def make_corpus(rng, n):
    out = []
    for _ in range(n):
        t = TEMPLATES[rng.integers(len(TEMPLATES))]
        p, q = rng.choice(PLACES, size=2, replace=False)
        out.append(t.format(c=rng.choice(COLORS), o=rng.choice(OBJECTS), p=p, q=q))
    return out


def learn_bpe(sentences, n_merges):
    """BPE: 隣り合う 2 つの記号の組で、いちばん多いものを 1 つの記号にまとめる. これを繰り返す."""
    words = Counter(tuple(s) for s in sentences)
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, f in words.items():
            for a, b in zip(w[:-1], w[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        merges.append((a, b, cnt))
        new_words = Counter()
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] += f
        words = new_words
    return merges


def apply_bpe(sentence, merges):
    w = list(sentence)
    for a, b, _ in merges:
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(w[i])
                i += 1
        w = out
    return w


class Bigram:
    """p(次のトークン | 1 つ前のトークン) を、数え上げ（と少しの下駄）で求める."""

    def __init__(self, seqs, alpha):
        vocab = sorted({t for s in seqs for t in s} | {BOS, EOS})
        self.vocab = vocab
        self.idx = {t: i for i, t in enumerate(vocab)}
        V = len(vocab)
        counts = np.zeros((V, V))
        for s in seqs:
            toks = [BOS] + s + [EOS]
            for a, b in zip(toks[:-1], toks[1:]):
                counts[self.idx[a], self.idx[b]] += 1
        self.counts = counts
        self.P = (counts + alpha) / (counts + alpha).sum(axis=1, keepdims=True)

    def nll(self, seqs):
        """トークン 1 つあたりの負の対数尤度（交差エントロピー）."""
        total, n = 0.0, 0
        for s in seqs:
            toks = [BOS] + s + [EOS]
            for a, b in zip(toks[:-1], toks[1:]):
                pa = self.P[self.idx[a]]
                total -= np.log(pa[self.idx[b]] if b in self.idx else 1e-12)
                n += 1
        return total / n

    def sample(self, rng, temp, max_len=40):
        cur, out = BOS, []
        for _ in range(max_len):
            logp = np.log(self.P[self.idx[cur]])
            if temp == 0:
                nxt = int(np.argmax(logp))
            else:
                z = logp / temp
                p = np.exp(z - z.max())
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            cur = self.vocab[nxt]
            if cur == EOS:
                break
            out.append(cur)
        return "".join(out)


def unigram_nll(train, test):
    c = Counter(t for s in train for t in s + [EOS])
    tot = sum(c.values())
    vocab = len(c)
    total, n = 0.0, 0
    for s in test:
        for t in s + [EOS]:
            total -= np.log((c[t] + 0.01) / (tot + 0.01 * vocab))
            n += 1
    return total / n


def main():
    rng = np.random.default_rng(0)
    train = make_corpus(rng, 3000)
    test = make_corpus(rng, 300)
    print("文の例:", " / ".join(train[:3]))

    print("\n(1) 文字を 1 つずつトークンにする")
    chars = sorted({ch for s in train for ch in s})
    print(f"    語彙（文字の種類）{len(chars)} 個、1 文の平均の長さ {np.mean([len(s) for s in train]):.1f} トークン")

    print("\n(2) BPE でトークンを作る（まとめる操作を 60 回）")
    merges = learn_bpe(train, 60)
    print("    最初の 10 回にまとめた組: " + "、".join(f"「{a}」+「{b}」" for a, b, _ in merges[:10]))
    ex = train[0]
    print(f"    例「{ex}」→ " + " | ".join(apply_bpe(ex, merges)))
    bpe_train = [apply_bpe(s, merges) for s in train]
    bpe_test = [apply_bpe(s, merges) for s in test]
    vocab_bpe = {t for s in bpe_train for t in s}
    print(f"    語彙 {len(vocab_bpe)} 個、1 文の平均の長さ {np.mean([len(s) for s in bpe_train]):.1f} トークン")

    print("\n(3) 言語モデルを学習し、見ていない 300 文で測る（トークン 1 つあたり）")
    char_train = [list(s) for s in train]
    char_test = [list(s) for s in test]
    print("    方法          語彙   一様に当てる   ユニグラム   バイグラム   1 文あたり（バイグラム）")
    for name, tr, te in (("文字", char_train, char_test), ("BPE ", bpe_train, bpe_test)):
        V = len({t for s in tr for t in s}) + 1
        u = unigram_nll(tr, te)
        bg = Bigram(tr, alpha=ALPHA)
        b = bg.nll(te)
        per_sentence = b * np.mean([len(s) + 1 for s in te])
        print(f"    {name}        {V:4d}   {np.log(V):12.2f}   {u:10.2f}   {b:10.2f}   {per_sentence:10.2f}")
    print("    （単位はナット. 一様に当てると log(語彙の数). 小さいほど、よく当てている）")
    bg = Bigram(bpe_train, alpha=ALPHA)
    row = bg.counts[bg.idx["ブロック"]]
    top = [j for j in np.argsort(-row) if row[j] > 0][:5]
    tot = row.sum()
    print("    数え上げの例: 「ブロック」の次に来たトークン（多い順に 5 つ）→ " +
          "、".join(f"「{bg.vocab[j]}」{int(row[j])} 回（{row[j] / tot:.2f}）" for j in top))

    print("\n(4) 学習した言語モデルから文を作る（BPE、バイグラム）")
    for temp in (0.0, 1.0, 3.0):
        r = np.random.default_rng(5)
        outs = [bg.sample(r, temp) for _ in range(3)]
        label = "温度 0（いちばん確率の高いものを選ぶ）" if temp == 0 else f"温度 {temp:.0f}"
        print(f"    {label}:")
        for o in (outs[:1] if temp == 0 else outs):
            print(f"      {o}")


if __name__ == "__main__":
    main()
