# -*- coding: utf-8 -*-
"""배포용 최종 모델 학습.

교차검증에서 만든 fold 모델들은 각각 데이터의 80%만 봤고 이미 삭제된다.
실제로 사람이 쓸 모델은 데이터 전체로 다시 한 번 학습해야 한다.
조기종료용 검증셋(10%)만 떼고 나머지 전부를 쓴다.

주의: 여기서 나오는 검증 정확도는 '성능 보고용 수치가 아니다.'
      논문에 쓸 수치는 finetune.py + report.py 의 out-of-fold 결과다.
      이 검증셋은 조기종료 시점을 정하는 데만 쓰였으므로 낙관적으로 나온다.

사용법:
    python train_final.py --model klue/roberta-base --out final_model
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from common import load_data, metrics
from finetune import TextDataset, build_args, load_tokenizer, make_trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--data", default="data_clean.csv")
    ap.add_argument("--out", default="final_model")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tok", choices=["auto", "plain"], default="auto")
    args = ap.parse_args()
    args.seed_run = args.seed
    args.tag = "final"

    from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed

    df, labels = load_data(args.data)
    set_seed(args.seed)
    y = df["label"].values
    texts = df["text"].values

    # 검증셋 10%: 유사 중복 그룹이 학습셋과 겹치지 않게 그룹 단위로 자른다.
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    tr_idx, va_idx = next(sgkf.split(df, y, groups=df["group_id"]))
    print("학습 %d편 / 검증 %d편 / 클래스 %s" % (len(tr_idx), len(va_idx), labels))

    tok, used_mecab = load_tokenizer(args.model, args.tok)
    tr_ds = TextDataset(texts[tr_idx], y[tr_idx], tok, args.max_len)
    va_ds = TextDataset(texts[va_idx], y[va_idx], tok, args.max_len)

    cnt = np.bincount(y[tr_idx], minlength=len(labels)).astype(np.float64)
    cw = torch.tensor(cnt.sum() / (len(labels) * np.maximum(cnt, 1)), dtype=torch.float)

    # id2label 을 넣어 두면 배포된 UI가 숫자 대신 분야 이름을 바로 보여준다.
    cfg = AutoConfig.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=cfg)

    targs = build_args(os.path.join("runs", "final"), args, max(1, len(tr_ds) // args.bs))
    trainer = make_trainer(model, targs, tr_ds, va_ds, tok, cw, args.patience)
    trainer.train()

    pred = trainer.predict(va_ds).predictions.argmax(-1)
    m = metrics(y[va_idx], pred)
    print("\n[검증셋] acc=%.4f macroF1=%.4f  (조기종료용 수치 — 논문에 인용하지 말 것)"
          % (m["accuracy"], m["macro_f1"]))

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    shutil.rmtree(os.path.join("runs", "final"), ignore_errors=True)

    print("저장 완료: %s" % os.path.abspath(args.out))
    if not used_mecab and "korscideberta" in args.model:
        print("주의: 대체 토크나이저로 학습됨. 배포 시에도 동일 조건이어야 한다.")
    print("\n다음: python export_web.py --model %s --out web/model" % args.out)


if __name__ == "__main__":
    main()
