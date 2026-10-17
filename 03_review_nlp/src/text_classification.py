"""
Topic classification of support-style text (20 Newsgroups, 20 classes, 18.8k docs)
plus unsupervised topic discovery with NMF.

The point of the project is the leakage experiment: the usual 20-Newsgroups
benchmark leaves e-mail headers, signatures and quoted replies in the text, which
leak the label. The same pipeline is trained twice - with and without that
metadata - and the gap between the two scores is reported, because the honest
number is the one without it.

Run:  python 03_review_nlp/src/text_classification.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common.viz import PALETTE, save, set_style

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "reports" / "figures"
SEED = 42
N_TOPICS = 15


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # drop pure numbers and 1-char tokens
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
    )


def candidates() -> dict[str, Pipeline]:
    return {
        "complement_nb": Pipeline([("tfidf", vectorizer()), ("clf", ComplementNB(alpha=0.3))]),
        "linear_svc": Pipeline([("tfidf", vectorizer()),
                                ("clf", LinearSVC(C=0.5, random_state=SEED))]),
        "logistic_regression": Pipeline([("tfidf", vectorizer()),
                                         ("clf", LogisticRegression(C=8.0, max_iter=1500,
                                                                    n_jobs=-1, random_state=SEED))]),
    }


def load_split(remove_metadata: bool):
    remove = ("headers", "footers", "quotes") if remove_metadata else ()
    tr = fetch_20newsgroups(subset="train", remove=remove, random_state=SEED)
    te = fetch_20newsgroups(subset="test", remove=remove, random_state=SEED)
    return tr, te


def main() -> None:
    t0 = time.time()
    print("[1/6] data (official train/test split, 20 classes)")
    tr_raw, te_raw = load_split(remove_metadata=False)
    tr, te = load_split(remove_metadata=True)
    names = list(tr.target_names)
    print("  train %d docs, test %d docs, %d classes" % (len(tr.data), len(te.data), len(names)))
    print("  median doc length (clean): %d characters"
          % int(np.median([len(t) for t in tr.data])))

    print("[2/6] leakage experiment: same pipeline, metadata kept vs removed")
    leak = {}
    for label, (a, b) in {"with_metadata": (tr_raw, te_raw), "without_metadata": (tr, te)}.items():
        pipe = candidates()["linear_svc"].fit(a.data, a.target)
        pred = pipe.predict(b.data)
        leak[label] = {
            "accuracy": float(accuracy_score(b.target, pred)),
            "macro_f1": float(f1_score(b.target, pred, average="macro")),
            "n_features": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        }
        print("  %-17s accuracy %.4f  macro-F1 %.4f  (%d tf-idf features)"
              % (label, leak[label]["accuracy"], leak[label]["macro_f1"], leak[label]["n_features"]))
    drop = leak["with_metadata"]["macro_f1"] - leak["without_metadata"]["macro_f1"]
    print("  -> headers/footers/quotes inflate macro-F1 by %.3f absolute" % drop)

    print("[3/6] model comparison on the clean corpus")
    rows = []
    fitted = {}
    for name, pipe in candidates().items():
        s = time.time()
        pipe.fit(tr.data, tr.target)
        pred = pipe.predict(te.data)
        fitted[name] = (pipe, pred)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(te.target, pred)),
            "macro_f1": float(f1_score(te.target, pred, average="macro")),
            "fit_seconds": round(time.time() - s, 1),
        })
        print("  %-20s accuracy %.4f  macro-F1 %.4f  (%.1fs)"
              % (name, rows[-1]["accuracy"], rows[-1]["macro_f1"], rows[-1]["fit_seconds"]))
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best_name = table.iloc[0]["model"]
    best, best_pred = fitted[best_name]
    print("  -> best: %s" % best_name)

    print("[4/6] per-class report, confusions and learning curve")
    report = classification_report(te.target, best_pred, target_names=names, output_dict=True)
    cm = confusion_matrix(te.target, best_pred)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [
        {"true": names[i], "predicted": names[j], "n": int(off[i, j])}
        for i, j in zip(*np.unravel_index(np.argsort(off.ravel())[::-1][:8], off.shape))
    ]
    for p in pairs[:5]:
        print("  confused %-26s -> %-26s %3d times" % (p["true"], p["predicted"], p["n"]))

    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(tr.data))
    curve = []
    for frac in sizes:
        k = int(len(idx) * frac)
        sub = idx[:k]
        pipe = candidates()[best_name].fit([tr.data[i] for i in sub], tr.target[sub])
        f1 = float(f1_score(te.target, pipe.predict(te.data), average="macro"))
        curve.append({"train_fraction": frac, "n_docs": k, "macro_f1": f1})
        print("  learning curve: %4d docs -> macro-F1 %.4f" % (k, f1))

    print("[5/6] unsupervised topics (NMF, %d components)" % N_TOPICS)
    tfidf = vectorizer()
    X = tfidf.fit_transform(tr.data)
    nmf = NMF(n_components=N_TOPICS, init="nndsvd", random_state=SEED, max_iter=400)
    W = nmf.fit_transform(X)
    vocab = np.array(tfidf.get_feature_names_out())
    topics = []
    for k, comp in enumerate(nmf.components_):
        top = vocab[np.argsort(comp)[::-1][:10]]
        dominant = pd.Series(tr.target[W.argmax(axis=1) == k]).map(lambda i: names[i])
        label = dominant.value_counts().index[0] if len(dominant) else "n/a"
        topics.append({"topic": k, "top_terms": list(top), "dominant_newsgroup": label,
                       "n_docs": int((W.argmax(axis=1) == k).sum())})
        print("  topic %2d [%-24s] %s" % (k, label, ", ".join(top[:7])))

    print("[6/6] figures + artefacts")
    set_style()
    per_class = pd.Series({n: report[n]["f1-score"] for n in names}).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(per_class.index, per_class.values, color=PALETTE[0])
    ax.axvline(report["macro avg"]["f1-score"], color=PALETTE[4], ls="--",
               label="macro avg %.3f" % report["macro avg"]["f1-score"])
    ax.set(title="Per-class F1 (%s, metadata removed)" % best_name, xlabel="F1")
    ax.legend()
    save(fig, FIG / "01_per_class_f1.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(20), names, rotation=90, fontsize=7)
    ax.set_yticks(range(20), names, fontsize=7)
    ax.set(title="Row-normalised confusion matrix", xlabel="predicted", ylabel="true")
    ax.grid(False)
    fig.colorbar(im, fraction=0.046)
    save(fig, FIG / "02_confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["with\nmetadata", "without\nmetadata"],
                [leak["with_metadata"]["macro_f1"], leak["without_metadata"]["macro_f1"]],
                color=[PALETTE[5], PALETTE[0]])
    for i, v in enumerate([leak["with_metadata"]["macro_f1"], leak["without_metadata"]["macro_f1"]]):
        axes[0].text(i, v + 0.01, "%.3f" % v, ha="center")
    axes[0].set(title="Metadata leakage inflates the benchmark", ylabel="macro-F1", ylim=(0, 1))
    c = pd.DataFrame(curve)
    axes[1].plot(c.n_docs, c.macro_f1, marker="o", color=PALETTE[1])
    axes[1].set(title="Learning curve (%s)" % best_name, xlabel="training documents",
                ylabel="macro-F1 on test")
    save(fig, FIG / "03_leakage_and_learning_curve.png")

    fig, axes = plt.subplots(3, 5, figsize=(16, 8))
    for k, ax in enumerate(axes.ravel()):
        comp = nmf.components_[k]
        order = np.argsort(comp)[::-1][:8][::-1]
        ax.barh(vocab[order], comp[order], color=PALETTE[k % len(PALETTE)])
        ax.set_title("topic %d: %s" % (k, topics[k]["dominant_newsgroup"]), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("NMF topics discovered without labels (top terms per topic)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, FIG / "04_nmf_topics.png")

    # most informative terms for a few classes, straight from the linear model
    # Term-level interpretability is read off the linear SVM, whose per-class weights are
    # directly comparable; ComplementNB log-probabilities are not (they are fitted on the
    # complement of each class, so the ranking is dominated by global term frequency).
    informative = {}
    svc, _ = fitted["linear_svc"]
    vocab_b = np.array(svc.named_steps["tfidf"].get_feature_names_out())
    coefs = svc.named_steps["clf"].coef_
    for cls in ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]:
        if cls in names:
            row = coefs[names.index(cls)]
            informative[cls] = list(vocab_b[np.argsort(row)[::-1][:12]])

    metrics = {
        "dataset": {
            "name": "20 Newsgroups (official train/test split)",
            "n_train": len(tr.data), "n_test": len(te.data), "n_classes": len(names),
        },
        "leakage_experiment": leak,
        "macro_f1_inflation_from_metadata": round(float(drop), 4),
        "model_comparison_clean": rows,
        "best_model": best_name,
        "best_model_test": {
            "accuracy": float(accuracy_score(te.target, best_pred)),
            "macro_f1": float(f1_score(te.target, best_pred, average="macro")),
            "weighted_f1": float(f1_score(te.target, best_pred, average="weighted")),
        },
        "worst_classes_f1": per_class.head(5).round(3).to_dict(),
        "best_classes_f1": per_class.tail(5).round(3).to_dict(),
        "top_confusions": pairs,
        "learning_curve": curve,
        "nmf_topics": topics,
        "most_informative_terms": informative,
    }
    (PROJ / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(PROJ / "reports" / "model_comparison.csv", index=False)

    wrong = np.where(best_pred != te.target)[0][:200]
    pd.DataFrame({
        "true": [names[te.target[i]] for i in wrong],
        "predicted": [names[best_pred[i]] for i in wrong],
        "text": [re.sub(r"\s+", " ", te.data[i])[:300] for i in wrong],
    }).to_csv(PROJ / "reports" / "misclassified_examples.csv", index=False)

    print("done in %.1fs | best %s macro-F1 %.3f on clean text; the same LinearSVC pipeline "
          "scores %.3f once header/footer metadata is left in"
          % (time.time() - t0, best_name, metrics["best_model_test"]["macro_f1"],
             leak["with_metadata"]["macro_f1"]))


if __name__ == "__main__":
    main()
