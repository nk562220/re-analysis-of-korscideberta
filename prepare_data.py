# -*- coding: utf-8 -*-
"""
1단계: 원본 크롤링 CSV -> 학습/평가용 정제 데이터셋(data_clean.csv)

이 단계의 목표는 '정확도를 높이는 것'이 아니라 '정확도를 믿을 수 있게 만드는 것'이다.
다음 세 가지를 처리한다.

  (1) 지도논문/학생논문 중복 병합
      과학전람회 데이터 2,349편 중 880편이 지도교사가 쓴 지도논문이다.
      본문(서론)이 학생논문과 거의 같으므로, 이 둘이 train/test로 갈라지면
      모델이 '외운 답'을 맞히게 되어 정확도가 부풀려진다(데이터 누수).

  (2) 크롤링 실패 행 제거
      요약문이 제목으로 채워졌거나 극단적으로 짧은 행(서론 40자 미만)을 제거한다.

  (3) group_id 부여
      (1)로 잡히지 않는 유사 중복까지 char n-gram 코사인 유사도로 묶어
      같은 그룹은 항상 같은 fold에 들어가게 한다(StratifiedGroupKFold용).

사용법:
    python prepare_data.py --sci "..\데이터\과학전람회데이터수정.csv" --out data_clean.csv
"""
import argparse
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

# 과학전람회 데이터의 분야 라벨(6종). '기타'는 내용상 하나의 주제가 아니어서
# 기본적으로 제외한다(--keep-etc 로 포함 가능).
FIELDS = ["물리", "화학", "생물", "지구및환경", "산업및에너지", "기타"]


def read_csv_any(path):
    """BOM/인코딩이 섞여 있어도 읽히도록 순차 시도."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("인코딩을 판별할 수 없습니다: %s" % path)


def norm_text(s):
    """전각 문자·중복 공백 등 크롤링 잡음 정리."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("，", ", ").replace("．", ". ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


TEACHER_PAT = re.compile(r"\(\s*지도\s*논문\s*\)|\[\s*지도\s*논문\s*\]")
# 지도논문 제목에 붙는 상투적 표현. 학생논문 제목과 매칭하기 전에 제거한다.
GUIDE_SUFFIX = re.compile(
    r"(에\s*대한|에\s*관한|을\s*위한|를\s*위한)?\s*"
    r"(연구\s*)?(지도|지도\s*과정|지도\s*사례|지도\s*방안|지도\s*논문)\s*$"
)


def clean_title(t):
    t = TEACHER_PAT.sub(" ", t)
    t = norm_text(t)
    t = GUIDE_SUFFIX.sub("", t).strip(" -–—·,.")
    return t


def merge_teacher_papers(df, title_thr, intro_thr):
    """지도논문을 같은 (연도, 분야) 안의 학생논문에 병합한다.

    반환: (병합된 df, 로그 dict)
    """
    from rapidfuzz import fuzz, process

    df = df.copy()
    df["is_teacher"] = df["title_raw"].apply(lambda t: bool(TEACHER_PAT.search(t)))

    stu = df[~df["is_teacher"]]
    tea = df[df["is_teacher"]]

    # 학생논문 후보를 (연도, 분야)별로 미리 묶어 둔다 -> 잘못된 매칭 억제 + 속도
    buckets = {}
    for idx, row in stu.iterrows():
        buckets.setdefault((row["year"], row["field"]), []).append(idx)

    merged_into = {}   # teacher idx -> student idx
    appended = 0
    for t_idx, t_row in tea.iterrows():
        cand = buckets.get((t_row["year"], t_row["field"]), [])
        if not cand:
            continue
        titles = [df.at[i, "title"] for i in cand]
        hit = process.extractOne(
            t_row["title"], titles, scorer=fuzz.token_set_ratio, score_cutoff=title_thr
        )
        if hit is None:
            continue
        s_idx = cand[hit[2]]
        merged_into[t_idx] = s_idx

        # 서론 병합: 내용이 사실상 같으면 학생 서론만 쓰고, 다르면 뒤에 덧붙인다.
        s_intro, t_intro = df.at[s_idx, "abstract"], t_row["abstract"]
        if fuzz.token_set_ratio(s_intro, t_intro) < intro_thr:
            df.at[s_idx, "abstract"] = (s_intro + " " + t_intro).strip()
            appended += 1

    df["merged_teacher"] = False
    df.loc[list(merged_into.keys()), "merged_teacher"] = True

    log = {
        "teacher_total": int(len(tea)),
        "teacher_matched": len(merged_into),
        "teacher_unmatched": int(len(tea)) - len(merged_into),
        "intro_appended": appended,
    }
    # 매칭된 지도논문 행은 삭제(학생논문에 흡수됨).
    # 매칭 안 된 지도논문은 대응 학생논문이 아예 없는 경우가 많아 독립 논문으로 남긴다.
    df = df[~df["merged_teacher"]].drop(columns=["merged_teacher"])
    return df, log


def assign_groups(texts, thr=0.90):
    """유사 중복 문서를 하나의 group_id로 묶는다(union-find).

    char_wb 3-5gram TF-IDF 코사인 유사도 >= thr 이면 같은 그룹.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    n = X.shape[0]

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 메모리를 위해 행 블록 단위로 유사도 계산
    step = 256
    pairs = 0
    for s in range(0, n, step):
        sim = (X[s:s + step] @ X.T).toarray()
        for r in range(sim.shape[0]):
            i = s + r
            sim[r, :i + 1] = 0.0            # 상삼각만
            for j in np.nonzero(sim[r] >= thr)[0]:
                union(i, int(j))
                pairs += 1

    groups = [find(i) for i in range(n)]
    remap = {g: k for k, g in enumerate(sorted(set(groups)))}
    return [remap[g] for g in groups], pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sci", required=True, help="과학전람회 크롤링 CSV")
    ap.add_argument("--rne", default=None, help="R&E 크롤링 CSV(분야 체계가 달라 학습에서 제외, 별도 저장)")
    ap.add_argument("--out", default="data_clean.csv")
    ap.add_argument("--min-chars", type=int, default=40, help="서론 최소 길이")
    ap.add_argument("--title-thr", type=int, default=60, help="지도-학생 제목 매칭 임계값")
    ap.add_argument("--intro-thr", type=int, default=60, help="서론 동일 판정 임계값")
    ap.add_argument("--dup-thr", type=float, default=0.90, help="유사 중복 그룹화 코사인 임계값")
    ap.add_argument("--keep-etc", action="store_true", help="'기타' 분야를 학습에 포함")
    args = ap.parse_args()

    log = {}
    df = read_csv_any(args.sci)
    log["rows_raw"] = len(df)

    df = df.rename(columns={"introduction": "abstract"})
    df["title_raw"] = df["title"].astype(str)
    df["title"] = df["title_raw"].apply(clean_title)
    df["abstract"] = df["abstract"].apply(norm_text)
    df["field"] = df["field"].astype(str).str.strip()
    df["year"] = df["year"].astype(str).str.slice(0, 4)

    # 서론 결측/공백 제거
    before = len(df)
    df = df[df["abstract"].str.len() > 0]
    log["drop_empty_abstract"] = before - len(df)

    # 완전 중복 행 제거(크롤링 페이지 경계에서 발생)
    before = len(df)
    df = df.drop_duplicates(subset=["title_raw", "abstract"])
    log["drop_exact_dup"] = before - len(df)

    # 지도논문 병합
    df, mlog = merge_teacher_papers(df, args.title_thr, args.intro_thr)
    log.update(mlog)

    # 크롤링 실패 행 제거: 서론이 제목과 같거나 너무 짧은 경우
    before = len(df)
    bad = (df["abstract"].str.len() < args.min_chars) | (
        df.apply(lambda r: r["abstract"][:25] == r["title"][:25] and len(r["abstract"]) < 120, axis=1)
    )
    log["drop_low_quality"] = int(bad.sum())
    df = df[~bad]

    # 라벨 정리
    df = df[df["field"].isin(FIELDS)]
    if not args.keep_etc:
        before = len(df)
        df = df[df["field"] != "기타"]
        log["drop_etc"] = before - len(df)

    df = df.reset_index(drop=True)
    df["text"] = (df["title"] + " [SEP] " + df["abstract"]).str.strip()

    labels = sorted(df["field"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    df["label"] = df["field"].map(l2i)

    # 유사 중복 그룹화(누수 차단용)
    df["group_id"], npairs = assign_groups(df["text"].tolist(), thr=args.dup_thr)
    log["near_dup_pairs"] = npairs
    log["n_groups"] = df["group_id"].nunique()

    df.insert(0, "id", np.arange(len(df)))
    cols = ["id", "title", "abstract", "text", "field", "label", "year", "group_id", "is_teacher"]
    df[cols].to_csv(args.out, index=False, encoding="utf-8-sig")

    with open(os.path.splitext(args.out)[0] + "_labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # ---- 요약 출력 ----
    print("=" * 62)
    print("전처리 로그")
    for k, v in log.items():
        print("  %-22s %s" % (k, v))
    print("-" * 62)
    print("최종 문서 수: %d,  그룹 수: %d,  클래스 수: %d" % (len(df), log["n_groups"], len(labels)))
    print("라벨 분포:")
    vc = df["field"].value_counts()
    for k in labels:
        print("  %-10s %4d  (%.1f%%)" % (k, vc[k], 100.0 * vc[k] / len(df)))
    ln = df["text"].str.len()
    print("본문 길이(문자): 평균 %.0f / 중앙 %.0f / 95%% %.0f / 최대 %d"
          % (ln.mean(), ln.median(), ln.quantile(0.95), ln.max()))
    print("→ 권장 max_len(토큰): %d" % int(min(512, 64 + ln.quantile(0.99) / 1.7)))
    print("연도 분포: %s" % df["year"].value_counts().sort_index().to_dict())
    print("저장: %s" % os.path.abspath(args.out))
    print("=" * 62)

    if args.rne:
        r = read_csv_any(args.rne)
        r = r.rename(columns={"introduction": "abstract"})
        r["title"] = r["title"].astype(str).apply(norm_text)
        r["abstract"] = r["abstract"].apply(norm_text)
        r["year"] = r["year"].astype(str).str.slice(0, 4)
        r["text"] = (r["title"] + " [SEP] " + r["abstract"]).str.strip()
        out2 = os.path.splitext(args.out)[0] + "_rne.csv"
        r[["title", "abstract", "text", "field", "year"]].to_csv(out2, index=False, encoding="utf-8-sig")
        print("R&E 데이터(분야 체계 상이, 학습 제외) 저장: %s  (%d편)" % (out2, len(r)))


if __name__ == "__main__":
    sys.exit(main())
