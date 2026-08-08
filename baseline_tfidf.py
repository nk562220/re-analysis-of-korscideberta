# -*- coding: utf-8 -*-
"""고전 베이스라인: TF-IDF + 선형분류기.

이 점수가 왜 필요한가:
  "KorSciDeBERTa 정확도 0.83"만 있으면 그 수치가 좋은지 나쁜지 알 수 없다.
  단어 빈도만 쓰는 30초짜리 모델이 0.80을 낸다면, 사전학습 모델의 기여는 3%p다.
  논문의 '성능 평가'는 이 비교 없이는 성립하지 않는다.

출력: oof/oof_tfidf_seed{seed}.npy  (전체 문서 x 클래스 확률, out-of-fold)
사용법:
    python baseline_tfidf.py --seeds 42 43 44
"""
import argparse
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from common import load_data, make_folds, metrics


def build_model(kind):
    # 한국어는 형태소 분석 없이도 char n-gram이 강하다. word와 함께 쓴다.
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    if kind == "svm":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    else:
        clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("feats", feats), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_clean.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--kind", choices=["logreg", "svm"], default="logreg")
    ap.add_argument("--outdir", default="oof")
    args = ap.parse_args()

    df, labels = load_data(args.data)
    os.makedirs(args.outdir, exist_ok=True)
    y = df["label"].values
    X = df["text"].values
    tag = "tfidf-%s" % args.kind

    for seed in args.seeds:
        fold = make_folds(df, seed)
        oof = np.zeros((len(df), len(labels)))
        for k in range(fold.max() + 1):
            tr, va = fold != k, fold == k
            model = build_model(args.kind)
            model.fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])
        m = metrics(y, oof.argmax(1))
        np.save(os.path.join(args.outdir, "oof_%s_seed%d.npy" % (tag, seed)), oof)
        print("[%s seed=%d] acc=%.4f  macroF1=%.4f" % (tag, seed, m["accuracy"], m["macro_f1"]))

    print("저장 완료 -> %s/oof_%s_seed*.npy" % (args.outdir, tag))


if __name__ == "__main__":
    main()
