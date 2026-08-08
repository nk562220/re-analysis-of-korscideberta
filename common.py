# -*- coding: utf-8 -*-
"""공통 유틸: 폴드 생성, 지표 계산, 부트스트랩 신뢰구간.

모든 모델이 '완전히 동일한 폴드'를 쓰도록 폴드 생성을 여기 한 곳에 모아 둔다.
모델마다 다른 분할을 쓰면 성능 차이가 모델 때문인지 분할 운 때문인지 알 수 없다.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

N_SPLITS = 5


def load_data(path="data_clean.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    labels = sorted(df["field"].unique())
    return df, labels


def make_folds(df, seed, n_splits=N_SPLITS):
    """유사 중복 그룹이 절대 분할을 넘나들지 않는 층화 K-fold.

    반환: fold 번호 배열(길이 = len(df))
    """
    fold = np.full(len(df), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(df, df["label"], groups=df["group_id"])):
        fold[va] = k
    assert (fold >= 0).all()
    return fold


def inner_split(df_tr, seed):
    """train fold 안에서 조기종료용 검증셋(10%)을 떼어낸다."""
    idx_tr, idx_va = train_test_split(
        np.arange(len(df_tr)), test_size=0.1, random_state=seed, stratify=df_tr["label"].values
    )
    return idx_tr, idx_va


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def bootstrap_ci(y_true, y_pred, groups, metric="accuracy", n_boot=2000, seed=0, alpha=0.05):
    """그룹 단위 리샘플링 부트스트랩 95% 신뢰구간.

    '정확도 0.83'만 쓰면 안 되는 이유: 표본이 1,431편뿐이라 ±2%p 정도는 우연이다.
    구간을 함께 보고해야 두 모델의 차이가 의미 있는지 판단할 수 있다.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    fn = accuracy_score if metric == "accuracy" else (lambda a, b: f1_score(a, b, average="macro"))

    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        vals[b] = fn(y_true[idx], y_pred[idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(y_true, pred_a, pred_b, groups, metric="macro_f1", n_boot=2000, seed=0):
    """두 모델 차이(A-B)의 부트스트랩 분포와 단측 p값.

    같은 데이터에 두 모델을 돌렸으므로 '짝지은' 비교를 해야 한다.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a, pred_b, groups = np.asarray(pred_a), np.asarray(pred_b), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    fn = accuracy_score if metric == "accuracy" else (lambda a, b: f1_score(a, b, average="macro"))

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        diffs[b] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
    obs = fn(y_true, pred_a) - fn(y_true, pred_b)
    p = float(np.mean(diffs <= 0)) if obs > 0 else float(np.mean(diffs >= 0))
    return obs, (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))), p


def text_report(y_true, y_pred, labels):
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    w = max(len(l) for l in labels) + 2
    lines = ["혼동행렬 (행=실제, 열=예측)", " " * w + "".join("%7s" % l[:6] for l in labels)]
    for i, l in enumerate(labels):
        lines.append("%-*s" % (w, l) + "".join("%7d" % v for v in cm[i]))
    return rep + "\n" + "\n".join(lines)
