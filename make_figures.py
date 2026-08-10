# -*- coding: utf-8 -*-
"""발표·논문용 그림 생성.

수치는 results/report.md 의 실측값이다(시드 42 OOF, 1,431편).
oof/*.npy 가 모두 있는 환경에서는 --from-oof 로 다시 계산할 수 있다.

사용법:
    python make_figures.py                 # 기록된 수치로 생성
    python make_figures.py --from-oof      # oof/*.npy 에서 재계산
"""
import argparse
import os

import numpy as np

NAVY, ORANGE, GRAY = "#1B365D", "#E8833A", "#8A94A6"
LABELS = ["물리", "산업및에너지", "생물", "지구및환경", "화학"]

# results/report.md 기준 (시드 42, out-of-fold 1,431편)
SCORES = {                       # (accuracy, macro_f1)
    "최빈 클래스": (0.2502, 0.0800),
    "TF-IDF": (0.6054, 0.5896),
    "KorSciDeBERTa": (0.6233, 0.6104),
    "KLUE-RoBERTa": (0.6730, 0.6625),
}
CONFUSION = np.array([          # KLUE-RoBERTa, 행=실제 열=예측
    [258, 32, 6, 16, 15],
    [21, 94, 12, 13, 16],
    [8, 14, 223, 64, 49],
    [21, 17, 39, 178, 35],
    [21, 9, 31, 29, 210],
])
FOLD_ACC = [0.6493, 0.7038, 0.7148, 0.6446, 0.6526]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "axes.edgecolor": GRAY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
    })
    return plt


def recompute_from_oof():
    """oof/*.npy 가 있으면 실제 예측에서 다시 계산한다."""
    import glob
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    from common import load_data, make_folds

    df, labels = load_data("data_clean.csv")
    y = df["label"].values
    out = {}
    for tag in ["tfidf-logreg", "korscideberta", "klue-roberta"]:
        fs = sorted(glob.glob("oof/oof_%s_seed*.npy" % tag))
        if not fs:
            print("  건너뜀(파일 없음): %s" % tag)
            continue
        p = np.mean([np.load(f) for f in fs], axis=0).argmax(1)
        out[tag] = (accuracy_score(y, p), f1_score(y, p, average="macro"))
        if tag == "klue-roberta":
            globals()["CONFUSION"] = confusion_matrix(y, p)
            fold = make_folds(df, 42)
            globals()["FOLD_ACC"] = [accuracy_score(y[fold == k], p[fold == k])
                                     for k in range(fold.max() + 1)]
    name = {"tfidf-logreg": "TF-IDF", "korscideberta": "KorSciDeBERTa",
            "klue-roberta": "KLUE-RoBERTa"}
    for t, v in out.items():
        SCORES[name[t]] = v
    print("  OOF 재계산 완료: %s" % list(out))


def fig_scores(plt, outdir):
    """모델별 성능 막대 — 발표의 핵심 슬라이드."""
    names = list(SCORES)
    acc = [SCORES[n][0] for n in names]
    f1 = [SCORES[n][1] for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color=NAVY)
    ax.bar(x + w / 2, f1, w, label="macro-F1", color=ORANGE)

    # 선행 연구가 보고한 값을 기준선으로 표시
    ax.axhline(0.0812, color="#C0392B", ls="--", lw=1.2)
    ax.text(1.15, 0.105, "선행 연구 보고값 0.0812 — 무작위 추측(0.20)보다 낮다",
            color="#C0392B", fontsize=8.5, ha="left")

    for xi, (a, f) in enumerate(zip(acc, f1)):
        ax.text(xi - w / 2, a + 0.012, "%.3f" % a, ha="center", fontsize=8, color=NAVY)
        ax.text(xi + w / 2, f + 0.012, "%.3f" % f, ha="center", fontsize=8, color=ORANGE)

    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("점수")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(outdir, "fig1_scores.png"))
    plt.close(fig)


def fig_confusion(plt, outdir):
    """혼동행렬 — 어디서 틀리는지가 고찰의 재료."""
    cm = CONFUSION
    row = cm.sum(1, keepdims=True)
    norm = cm / row

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.5 else NAVY)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=9)
    ax.set_xlabel("예측 분야")
    ax.set_ylabel("실제 분야")
    fig.colorbar(im, ax=ax, shrink=0.8, label="행 기준 비율")
    fig.savefig(os.path.join(outdir, "fig2_confusion.png"))
    plt.close(fig)


def fig_folds(plt, outdir):
    """겹별 변동 — 단일 분할로 재면 안 되는 이유."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(1, len(FOLD_ACC) + 1)
    ax.plot(x, FOLD_ACC, "o-", color=NAVY, lw=2, ms=7)
    mean = float(np.mean(FOLD_ACC))
    ax.axhline(mean, color=ORANGE, ls="--", lw=1.5)
    ax.text(len(FOLD_ACC) + 0.05, mean, " 전체 OOF\n %.4f" % mean,
            color=ORANGE, fontsize=9, va="center")
    for xi, v in zip(x, FOLD_ACC):
        ax.text(xi, v + 0.006, "%.3f" % v, ha="center", fontsize=8)

    lo, hi = min(FOLD_ACC), max(FOLD_ACC)
    ax.annotate("", xy=(1.35, hi), xytext=(1.35, lo),
                arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.3))
    ax.text(1.45, (lo + hi) / 2, "%.1f%%p 차이" % ((hi - lo) * 100),
            color="#C0392B", fontsize=9, va="center")

    ax.set_xticks(x, ["겹 %d" % i for i in x], fontsize=9)
    ax.set_ylim(lo - 0.03, hi + 0.035)
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.7, len(FOLD_ACC) + 0.75)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(outdir, "fig3_folds.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--from-oof", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.from_oof:
        print("OOF 파일에서 재계산 시도")
        recompute_from_oof()

    plt = setup()
    fig_scores(plt, args.outdir)
    fig_confusion(plt, args.outdir)
    fig_folds(plt, args.outdir)
    for f in sorted(os.listdir(args.outdir)):
        p = os.path.join(args.outdir, f)
        print("  %-24s %6.0f KB" % (f, os.path.getsize(p) / 1024))
    print("저장: %s" % os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
