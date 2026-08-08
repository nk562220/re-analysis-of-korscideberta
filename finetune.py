# -*- coding: utf-8 -*-
"""사전학습 모델 파인튜닝 + out-of-fold 예측 저장.

이전 프로젝트에서 정확도가 0.08로 나온 원인은 분류 헤드를 '학습하지 않고' 평가했기 때문이다
(AutoModelForSequenceClassification 는 무작위 초기화된 헤드를 함께 만든다).
이 스크립트는 실제로 학습 루프를 돌리고, 각 문서가 '학습에 쓰이지 않은 fold'에 있을 때의
예측만 모아서(out-of-fold) 전체 1,431편에 대한 단일 정확도를 계산한다.
단일 홀드아웃(예: 15%=215편)보다 표본이 6.6배 커서 훨씬 정밀한 수치가 나온다.

사용법(GPU 권장):
    python finetune.py --model kisti/korscideberta --tag korscideberta --seeds 42 43 44
    python finetune.py --model klue/roberta-base    --tag klue-roberta  --seeds 42 43 44

CPU에서도 동작하지만 1 fold에 수십 분~수 시간이 걸린다. --max-len 256 --bs 8 로 줄일 것.
"""
import argparse
import os
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from common import inner_split, load_data, make_folds, metrics

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------- 토크나이저
def load_tokenizer(model_name, mode="auto"):
    """KorSciDeBERTa 전용 토크나이저를 최대한 원본대로 불러온다.

    kisti/korscideberta 는 Mecab-ko 형태소 분석 후 SentencePiece 를 적용하는
    커스텀 토크나이저(tokenization_korscideberta_v2.py)를 쓴다.
    사전학습과 동일한 방식이라 성능상 이 경로가 가장 좋지만 mecab 설치가 필요하다.
    실패하면 같은 spm.model 을 쓰는 표준 DebertaV2Tokenizer 로 자동 강등한다
    (형태소 분리만 빠지므로 보통 1~2%p 손해, 실행은 됨).
    """
    from transformers import AutoTokenizer, DebertaV2Tokenizer

    if "korscideberta" in model_name and mode != "plain":
        try:
            from huggingface_hub import snapshot_download

            from mecab_compat import ensure_konlpy_mecab
            if ensure_konlpy_mecab() is None:
                raise RuntimeError("Mecab-ko 백엔드 없음")

            # 저장소에 korscideberta.zip(1.46GB)이 있어 전체 스냅샷을 받으면 안 된다.
            # 토크나이저에 필요한 파일만 내려받는다.
            path = snapshot_download(
                model_name,
                allow_patterns=[
                    "tokenization_korscideberta_v2.py", "unicode.py", "normalize.py",
                    "spm.model", "tokenizer.model", "vocab.txt",
                    "tokenizer_config.json", "config.json",
                ],
            )
            if path not in sys.path:
                sys.path.insert(0, path)   # unicode.py / normalize.py 를 찾게 함
            from tokenization_korscideberta_v2 import DebertaV2Tokenizer as KorSciTok
            tok = KorSciTok.from_pretrained(path)
            tok("토크나이저 동작 확인")      # 실제로 한 번 돌려서 검증
            print("[tokenizer] KorSciDeBERTa 커스텀 토크나이저(Mecab-ko) 사용")
            return tok, True
        except Exception as e:
            print("[tokenizer] 커스텀 토크나이저 실패(%s: %s)" % (type(e).__name__, e))
            print("[tokenizer] -> spm.model 기반 DebertaV2Tokenizer 로 대체")
            try:
                from huggingface_hub import hf_hub_download
                spm = hf_hub_download(model_name, "spm.model")
                return DebertaV2Tokenizer(vocab_file=spm), False
            except Exception as e2:
                print("[tokenizer] spm 직접 로드도 실패: %s" % e2)

    return AutoTokenizer.from_pretrained(model_name), False


class TextDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


# ---------------------------------------------------------------- 학습 인자
def build_args(outdir, args, steps_per_epoch):
    """transformers 버전마다 인자 이름이 바뀌므로(evaluation_strategy -> eval_strategy)
    한 번에 성공할 때까지 순차 시도한다. 지난번에 여기서 막혀 학습을 포기했다."""
    from transformers import TrainingArguments

    base = dict(
        output_dir=outdir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        logging_steps=max(10, steps_per_epoch // 2),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed_run,
        report_to=[],
        disable_tqdm=False,
        label_smoothing_factor=args.label_smoothing,
    )
    for key in ("eval_strategy", "evaluation_strategy"):
        try:
            return TrainingArguments(**base, **{key: "epoch"}, save_strategy="epoch")
        except TypeError:
            continue
    return TrainingArguments(**base)  # 최후 수단


def make_trainer(model, targs, tr_ds, va_ds, tok, class_weight, patience):
    from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer
    from sklearn.metrics import accuracy_score, f1_score

    class WeightedTrainer(Trainer):
        """클래스 불균형(산업및에너지 156편 vs 생물 358편) 보정.
        macro-F1 을 목표 지표로 쓰므로 소수 클래스에 가중치를 준다."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            out = model(**inputs)
            w = class_weight.to(out.logits.device) if class_weight is not None else None
            loss = nn.CrossEntropyLoss(weight=w, label_smoothing=targs.label_smoothing_factor)(
                out.logits, labels
            )
            return (loss, out) if return_outputs else loss

    def compute_metrics(p):
        pred = p.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(p.label_ids, pred),
            "macro_f1": f1_score(p.label_ids, pred, average="macro"),
        }

    kw = dict(
        model=model, args=targs, train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    # transformers 4.46+ 는 tokenizer= 가 deprecated, processing_class= 를 쓴다
    try:
        return WeightedTrainer(processing_class=tok, **kw)
    except TypeError:
        return WeightedTrainer(tokenizer=tok, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="결과 파일 이름에 쓰일 짧은 이름")
    ap.add_argument("--data", default="data_clean.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--tok", choices=["auto", "plain"], default="auto")
    ap.add_argument("--outdir", default="oof")
    ap.add_argument("--workdir", default="runs")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed

    df, labels = load_data(args.data)
    y = df["label"].values
    texts = df["text"].values
    os.makedirs(args.outdir, exist_ok=True)
    print("장치: %s / 문서 %d편 / 클래스 %d개" %
          ("cuda" if torch.cuda.is_available() else "cpu", len(df), len(labels)))

    tok, used_mecab = load_tokenizer(args.model, args.tok)

    for seed in args.seeds:
        args.seed_run = seed
        set_seed(seed)
        fold = make_folds(df, seed)
        oof = np.zeros((len(df), len(labels)))

        partdir = os.path.join(args.outdir, "parts")
        os.makedirs(partdir, exist_ok=True)

        for k in range(fold.max() + 1):
            tr_idx = np.nonzero(fold != k)[0]
            te_idx = np.nonzero(fold == k)[0]

            # Colab 은 세션이 끊기므로 fold 단위로 결과를 저장하고, 있으면 건너뛴다.
            part = os.path.join(partdir, "%s_s%d_f%d.npy" % (args.tag, seed, k))
            if os.path.exists(part):
                oof[te_idx] = np.load(part)
                m = metrics(y[te_idx], oof[te_idx].argmax(1))
                print("  [%s seed=%d fold=%d] 이미 완료됨 -> 건너뜀 (acc=%.4f)"
                      % (args.tag, seed, k, m["accuracy"]))
                continue

            df_tr = df.iloc[tr_idx]
            i_tr, i_va = inner_split(df_tr, seed)          # 조기종료용 내부 검증셋

            tr_ds = TextDataset(texts[tr_idx][i_tr], y[tr_idx][i_tr], tok, args.max_len)
            va_ds = TextDataset(texts[tr_idx][i_va], y[tr_idx][i_va], tok, args.max_len)
            te_ds = TextDataset(texts[te_idx], y[te_idx], tok, args.max_len)

            cw = None
            if not args.no_class_weight:
                cnt = np.bincount(y[tr_idx][i_tr], minlength=len(labels)).astype(np.float64)
                cw = torch.tensor(cnt.sum() / (len(labels) * np.maximum(cnt, 1)), dtype=torch.float)

            cfg = AutoConfig.from_pretrained(args.model, num_labels=len(labels))
            model = AutoModelForSequenceClassification.from_pretrained(args.model, config=cfg)

            outdir = os.path.join(args.workdir, "%s_s%d_f%d" % (args.tag, seed, k))
            targs = build_args(outdir, args, max(1, len(tr_ds) // max(1, args.bs)))
            trainer = make_trainer(model, targs, tr_ds, va_ds, tok, cw, args.patience)
            trainer.train()

            logits = trainer.predict(te_ds).predictions
            oof[te_idx] = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            np.save(part, oof[te_idx])

            m = metrics(y[te_idx], oof[te_idx].argmax(1))
            print("  [%s seed=%d fold=%d] acc=%.4f macroF1=%.4f"
                  % (args.tag, seed, k, m["accuracy"], m["macro_f1"]))
            del trainer, model
            torch.cuda.empty_cache()
            # 체크포인트는 fold마다 700MB 넘는다. Colab 디스크가 차지 않게 즉시 삭제.
            shutil.rmtree(outdir, ignore_errors=True)

        m = metrics(y, oof.argmax(1))
        path = os.path.join(args.outdir, "oof_%s_seed%d.npy" % (args.tag, seed))
        np.save(path, oof)
        print("[%s seed=%d] OOF acc=%.4f macroF1=%.4f -> %s"
              % (args.tag, seed, m["accuracy"], m["macro_f1"], path))

    if "korscideberta" in args.model and not used_mecab:
        print("\n주의: Mecab-ko 커스텀 토크나이저가 아닌 대체 경로로 실행되었습니다.")
        print("      논문에 '사전학습과 동일한 토크나이저를 사용했다'고 쓰지 말 것.")


if __name__ == "__main__":
    main()
