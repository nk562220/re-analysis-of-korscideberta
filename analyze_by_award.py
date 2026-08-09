# -*- coding: utf-8 -*-
"""수상 등급별 분류 정확도 분석.

연구 질문: 높은 상을 받은 논문일수록 제목·초록만으로 분야가 더 명확히 드러나는가?
  즉 '좋은 논문은 자기 주제를 더 분명하게 서술한다'는 가설을 정량적으로 검증한다.

재학습은 필요 없다. finetune.py 가 만든 out-of-fold 예측(oof/*.npy)에
등급만 붙여서 집단별로 비교한다. OOF 는 각 논문이 학습에 쓰이지 않은 상태의
예측이므로 이런 사후 분석에 그대로 쓸 수 있다.

입력:
  data_clean.csv            (prepare_data.py 산출물)
  oof/oof_<tag>_seed*.npy   (finetune.py 산출물)
  awards.csv                (title, year, award  — crawl 로 추가 수집해야 함)

사용법:
    python analyze_by_award.py --awards awards.csv --tag klue-roberta
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from common import load_data

# 전국과학전람회 등급을 순위로 매핑(숫자가 클수록 높은 상).
# 데이터에 없는 등급이 나오면 경고와 함께 목록을 출력하므로 --rank-map 으로 보완한다.
DEFAULT_RANKS = {
    "대통령상": 6,
    "국무총리상": 5,
    "최우수상": 4,
    "특상": 3,
    "우수상": 2,
    "장려상": 1,
}
# 지도교사/단체에게 주는 상은 논문 서술 품질과 무관하므로 제외한다.
EXCLUDE_PAT = re.compile(r"지도|단체|노력|참가|입선")


def norm_title(s):
    s = re.sub(r"\(\s*지도\s*논문\s*\)", " ", str(s))
    s = re.sub(r"[^\w가-힣]+", "", s)
    return s.lower()


def attach_awards(df, awards, title_thr):
    """awards.csv 를 data_clean.csv 에 붙인다. 같은 연도 안에서만 매칭한다."""
    from rapidfuzz import fuzz, process

    aw = awards.copy()
    aw["award"] = aw["award"].astype(str).str.strip()
    aw = aw[~aw["award"].str.contains(EXCLUDE_PAT, na=False)]
    aw["year"] = aw["year"].astype(str).str.slice(0, 4)
    aw["key"] = aw["title"].map(norm_title)

    df = df.copy()
    df["year"] = df["year"].astype(str).str.slice(0, 4)
    df["key"] = df["title"].map(norm_title)

    by_year = {y: g for y, g in aw.groupby("year")}
    out, exact, fuzzy, miss = [], 0, 0, 0
    for _, row in df.iterrows():
        g = by_year.get(row["year"])
        if g is None or len(g) == 0:
            out.append(None); miss += 1; continue
        hit = g[g["key"] == row["key"]]
        if len(hit):
            out.append(hit.iloc[0]["award"]); exact += 1; continue
        keys = g["key"].tolist()
        m = process.extractOne(row["key"], keys, scorer=fuzz.ratio, score_cutoff=title_thr)
        if m is None:
            out.append(None); miss += 1
        else:
            out.append(g.iloc[m[2]]["award"]); fuzzy += 1
    df["award"] = out
    print("등급 매칭: 정확 %d / 유사 %d / 실패 %d (총 %d)" % (exact, fuzzy, miss, len(df)))
    return df


def load_oof(outdir, tag, n, c):
    files = sorted(glob.glob(os.path.join(outdir, "oof_%s_seed*.npy" % tag)))
    if not files:
        raise SystemExit("OOF 파일이 없습니다: %s/oof_%s_seed*.npy" % (outdir, tag))
    probs = np.mean([np.load(f) for f in files], axis=0)
    assert probs.shape == (n, c), "OOF 크기 불일치: %s" % (probs.shape,)
    print("OOF: %s (시드 %d개 평균)" % (tag, len(files)))
    return probs


def bucketize(df, min_n):
    """표본이 너무 작은 등급을 인접 등급과 합친다(대통령상은 연 1~2편뿐)."""
    order = sorted(df["rank"].dropna().unique())
    counts = df["rank"].value_counts()
    groups, cur = [], []
    for r in order:                       # 낮은 등급부터 쌓다가 min_n 넘으면 확정
        cur.append(r)
        if counts[cur].sum() >= min_n:
            groups.append(list(cur)); cur = []
    if cur:                               # 남은 꼬리는 마지막 묶음에 붙인다
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    mapping, names = {}, {}
    rev = {v: k for k, v in DEFAULT_RANKS.items()}
    for i, g in enumerate(groups):
        label = "+".join(rev.get(r, str(r)) for r in sorted(g, reverse=True))
        for r in g:
            mapping[r] = i
            names[i] = label
    df["bucket"] = df["rank"].map(mapping)
    df["bucket_name"] = df["bucket"].map(names)
    return df


def permutation_spearman(x, y, n_perm=5000, seed=0):
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    yv = np.asarray(y)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(yv)).statistic
    p = float(np.mean(np.abs(null) >= abs(rho)))
    return float(rho), p


def controlled_logit(df, n_boot=1000, seed=0):
    """correct ~ 등급순위 + log(길이) + 분야  로지스틱 회귀.

    상위 수상작이 초록을 길게 쓰거나 특정 분야에 몰려 있을 수 있으므로
    그 두 요인을 통제한 뒤에도 등급 효과가 남는지 본다.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=["rank"]).copy()
    X = pd.get_dummies(d["field"], prefix="f", drop_first=True).astype(float)
    X["rank"] = d["rank"].values
    X["loglen"] = np.log(d["text"].str.len().values)
    cols = list(X.columns)
    Xv = StandardScaler().fit_transform(X.values)
    yv = d["correct"].values

    def fit(idx):
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Xv[idx], yv[idx])
        return m.coef_[0][cols.index("rank")]

    base = fit(np.arange(len(d)))
    rng = np.random.default_rng(seed)
    boots = np.array([fit(rng.integers(0, len(d), len(d))) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return base, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_clean.csv")
    ap.add_argument("--awards", required=True, help="title,year,award 컬럼을 가진 CSV")
    ap.add_argument("--tag", default="klue-roberta")
    ap.add_argument("--outdir", default="oof")
    ap.add_argument("--out", default="results/award_analysis.md")
    ap.add_argument("--title-thr", type=int, default=90)
    ap.add_argument("--min-n", type=int, default=60, help="등급 묶음 최소 표본")
    ap.add_argument("--rank-map", default=None, help='추가 등급 매핑 JSON 예: \'{"특별상":3}\'')
    args = ap.parse_args()

    ranks = dict(DEFAULT_RANKS)
    if args.rank_map:
        ranks.update(json.loads(args.rank_map))

    df, labels = load_data(args.data)
    probs = load_oof(args.outdir, args.tag, len(df), len(labels))

    y = df["label"].values
    pred = probs.argmax(1)
    df["correct"] = (pred == y).astype(int)
    df["p_true"] = probs[np.arange(len(df)), y]          # 정답 분야에 준 확률
    srt = np.sort(probs, axis=1)
    df["margin"] = srt[:, -1] - srt[:, -2]               # 1위와 2위의 확률 차 = 확신도

    awards = pd.read_csv(args.awards, encoding="utf-8-sig")
    need = {"title", "year", "award"}
    assert need <= set(awards.columns), "awards.csv 에 %s 컬럼이 필요합니다" % need
    df = attach_awards(df, awards, args.title_thr)

    unknown = sorted(set(df["award"].dropna()) - set(ranks))
    if unknown:
        print("\n※ 순위를 모르는 등급 %d종 — --rank-map 으로 지정하세요:" % len(unknown))
        for u in unknown:
            print("   %s (%d편)" % (u, int((df["award"] == u).sum())))

    df["rank"] = df["award"].map(ranks)
    d = df.dropna(subset=["rank"]).copy()
    if len(d) < 100:
        raise SystemExit("등급이 붙은 논문이 %d편뿐입니다. 매칭 임계값(--title-thr)이나 awards.csv 를 확인하세요." % len(d))
    d = bucketize(d, args.min_n)

    L = []
    A = L.append
    A("# 수상 등급별 분류 정확도\n")
    A("- 모델: %s (out-of-fold 예측, 재학습 없음)" % args.tag)
    A("- 등급이 확인된 논문: %d / %d편" % (len(d), len(df)))
    A("- 지표: `p_true` = 모델이 정답 분야에 준 확률, `margin` = 1위와 2위 확률의 차\n")

    A("## 1. 등급별 지표\n")
    A("| 등급 | n | Accuracy | 평균 p_true | 평균 margin | 평균 길이(자) |")
    A("|---|---|---|---|---|---|")
    for b, g in d.groupby("bucket"):
        A("| %s | %d | %.4f | %.4f | %.4f | %.0f |"
          % (g["bucket_name"].iloc[0], len(g), g["correct"].mean(),
             g["p_true"].mean(), g["margin"].mean(), g["text"].str.len().mean()))

    A("\n## 2. 등급과 예측 확신도의 상관 (개별 논문 단위)\n")
    A("| 종속변수 | Spearman rho | 순열검정 p |")
    A("|---|---|---|")
    for name, col in [("정답 여부", "correct"), ("p_true", "p_true"), ("margin", "margin")]:
        rho, p = permutation_spearman(d["rank"].values, d[col].values)
        A("| %s | %+.4f | %.4f |" % (name, rho, p))
    A("\nrho > 0 이면 '높은 상일수록 분야가 명확히 드러난다'는 가설과 부합한다.")
    A("p >= 0.05 면 이 표본으로는 방향을 단정할 수 없다.")

    A("\n## 3. 초록 길이와 분야 구성을 통제한 뒤의 등급 효과\n")
    coef, lo, hi = controlled_logit(d)
    A("로지스틱 회귀 `correct ~ 등급순위 + log(초록길이) + 분야`")
    A("")
    A("- 등급순위 계수: **%+.4f** (부트스트랩 95%% CI [%+.4f, %+.4f])" % (coef, lo, hi))
    A("- CI가 0을 포함하지 않으면, 길이·분야 차이로 설명되지 않는 등급 효과가 있다는 뜻이다.")
    A("")
    A("이 통제가 필요한 이유: 상위 수상작은 초록이 더 길거나 특정 분야에 몰릴 수 있고,")
    A("그 경우 등급이 아니라 길이나 분야가 정확도를 끌어올린 것일 수 있다.")

    A("\n## 4. 분야별 등급 효과 (교란 확인)\n")
    A("| 분야 | n | 최하위 등급군 Acc | 최상위 등급군 Acc | 차이 |")
    A("|---|---|---|---|---|")
    lo_b, hi_b = d["bucket"].min(), d["bucket"].max()
    for f, g in d.groupby("field"):
        a, b = g[g["bucket"] == lo_b], g[g["bucket"] == hi_b]
        if len(a) < 5 or len(b) < 5:
            A("| %s | %d | (표본 부족) | | |" % (f, len(g)))
            continue
        A("| %s | %d | %.3f (n=%d) | %.3f (n=%d) | %+.3f |"
          % (f, len(g), a["correct"].mean(), len(a), b["correct"].mean(), len(b),
             b["correct"].mean() - a["correct"].mean()))

    A("\n## 5. 해석 시 주의\n")
    A("- 수상 등급은 심사위원의 종합 평가이고, 초록 서술의 명확성은 그 일부일 뿐이다.")
    A("  상관이 나와도 '잘 쓴 초록이 상을 받게 했다'는 인과로 읽으면 안 된다.")
    A("- 상위 등급의 표본이 작다. 대통령상은 연 1~2편이라 인접 등급과 합쳐서 본다.")
    A("- 모델 정확도의 상한은 과학전람회 분야 배정 자체의 일관성이 결정한다.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    txt = "\n".join(L)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print("\n저장: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
