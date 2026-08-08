# -*- coding: utf-8 -*-
"""oof/*.npy 를 모아 최종 성능 표를 만든다.

보고하는 것:
  - 시드별 정확도/macro-F1 의 평균 ± 표준편차  (분할 운에 의한 흔들림 크기)
  - 시드 앙상블(확률 평균) 성능                (실제로 쓸 때의 최고 성능)
  - 그룹 부트스트랩 95% 신뢰구간               (표본 1,431편이 주는 불확실성)
  - 클래스별 P/R/F1 + 혼동행렬                 ('어디서 틀리는가'가 고찰의 재료)
  - 모델 간 짝지은 부트스트랩 비교              ('차이가 우연인가'에 대한 답)

사용법:
    python report.py                       # oof 폴더 전체
    python report.py --tags korscideberta tfidf-logreg
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

from common import bootstrap_ci, load_data, metrics, paired_bootstrap, text_report


def collect(outdir):
    """tag -> {seed: probs}"""
    res = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(outdir, "oof_*_seed*.npy"))):
        m = re.match(r"oof_(.+)_seed(\d+)\.npy$", os.path.basename(p))
        if m:
            res[m.group(1)][int(m.group(2))] = np.load(p)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_clean.csv")
    ap.add_argument("--outdir", default="oof")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--out", default="results/report.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df, labels = load_data(args.data)
    y = df["label"].values
    groups = df["group_id"].values

    res = collect(args.outdir)
    if args.tags:
        res = {t: v for t, v in res.items() if t in args.tags}
    if not res:
        raise SystemExit("oof/*.npy 가 없습니다. baseline_tfidf.py / finetune.py 를 먼저 실행하세요.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    L = []
    A = L.append
    A("# 청소년 논문 분야 분류 성능\n")
    A("- 데이터: %d편, %d클래스 %s" % (len(df), len(labels), labels))
    A("- 평가: 5-fold StratifiedGroupKFold out-of-fold 예측(전체 문서가 정확히 1번 평가됨)")
    A("- 유사 중복 문서는 같은 fold에 묶여 train/test 누수 없음\n")

    A("## 1. 모델별 성능\n")
    A("| 모델 | 시드 수 | Accuracy (mean±sd) | macro-F1 (mean±sd) | 앙상블 Acc | 앙상블 macro-F1 | Top-2 Acc | Acc 95% CI |")
    A("|---|---|---|---|---|---|---|---|")

    ens_pred, ens_prob = {}, {}
    for tag, per_seed in sorted(res.items()):
        accs, f1s = [], []
        for s, prob in sorted(per_seed.items()):
            m = metrics(y, prob.argmax(1))
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])
        prob_mean = np.mean(list(per_seed.values()), axis=0)
        pred = prob_mean.argmax(1)
        ens_pred[tag], ens_prob[tag] = pred, prob_mean
        me = metrics(y, pred)
        # Top-2 정확도: 분야 경계가 실제로 모호한 논문이 많아(예: 항균 소재 = 화학/생물)
        # 선행연구(박태진 2025)도 Top-k 를 함께 보고한다.
        top2 = float(np.mean([y[i] in np.argsort(-prob_mean[i])[:2] for i in range(len(y))]))
        lo, hi = bootstrap_ci(y, pred, groups, "accuracy", args.n_boot)
        A("| %s | %d | %.4f ± %.4f | %.4f ± %.4f | **%.4f** | **%.4f** | %.4f | [%.3f, %.3f] |"
          % (tag, len(per_seed), np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s),
             me["accuracy"], me["macro_f1"], top2, lo, hi))

    # 전체 모델 앙상블
    if len(ens_prob) > 1:
        allp = np.mean(list(ens_prob.values()), axis=0).argmax(1)
        m = metrics(y, allp)
        A("| (전체 앙상블) | - | - | - | **%.4f** | **%.4f** | - | - |" % (m["accuracy"], m["macro_f1"]))

    A("\n## 2. 참고 기준선\n")
    maj = np.full(len(y), np.bincount(y).argmax())
    mm = metrics(y, maj)
    A("- 최빈 클래스만 찍기: Accuracy %.4f / macro-F1 %.4f" % (mm["accuracy"], mm["macro_f1"]))
    A("- 무작위 추측(균등): Accuracy %.4f" % (1.0 / len(labels)))
    A("- 참고: 2025년 이전 연구의 0.0812 는 학습되지 않은 무작위 분류 헤드의 출력이므로")
    A("  위 기준선보다도 낮으며, 성능 수치로 인용할 수 없다.")

    A("\n## 3. 클래스별 성능 및 혼동행렬\n")
    for tag in sorted(ens_pred):
        A("### %s\n" % tag)
        A("```")
        A(text_report(y, ens_pred[tag], labels))
        A("```\n")

    if len(ens_pred) > 1:
        A("## 4. 모델 간 차이의 통계적 유의성 (짝지은 부트스트랩, macro-F1)\n")
        A("| A | B | A-B | 95% CI | p |")
        A("|---|---|---|---|---|")
        tags = sorted(ens_pred, key=lambda t: -metrics(y, ens_pred[t])["macro_f1"])
        best = tags[0]
        for t in tags[1:]:
            obs, ci, p = paired_bootstrap(y, ens_pred[best], ens_pred[t], groups,
                                          "macro_f1", args.n_boot)
            A("| %s | %s | %+.4f | [%+.4f, %+.4f] | %.4f |" % (best, t, obs, ci[0], ci[1], p))
        A("\nCI가 0을 포함하면 두 모델의 차이는 이 데이터 크기로는 단정할 수 없다.")

    A("\n## 5. 오분류 사례 (상위 모델, 확신도 높은 오답 10건)\n")
    top = sorted(ens_pred, key=lambda t: -metrics(y, ens_pred[t])["macro_f1"])[0]
    prob = ens_prob[top]
    wrong = np.nonzero(ens_pred[top] != y)[0]
    wrong = wrong[np.argsort(-prob[wrong].max(1))][:10]
    A("| 실제 | 예측 | 확신도 | 제목 |")
    A("|---|---|---|---|")
    for i in wrong:
        A("| %s | %s | %.2f | %s |" % (labels[y[i]], labels[ens_pred[top][i]],
                                       prob[i].max(), str(df["title"].iloc[i])[:50]))

    txt = "\n".join(L)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n저장: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
