# Colab GPU에서 돌리기

노트북(.ipynb)을 새로 만들 필요는 없다. 빈 Colab 노트에 아래 셀들을 붙여
`!python <스크립트>` 로 실행하면 된다. 코드는 전부 이 폴더의 `.py` 파일 그대로다.

**중요:** 이전 프로젝트를 막았던 "Python 3.8~3.9, torch 1.10" 제약은
KISTI 문서에 적힌 **원본 DeBERTa 학습 저장소용** 조건이다.
`transformers`로 모델을 불러 쓰는 지금 방식에서는 Colab 기본 파이썬·torch로 문제없다.
파이썬 버전을 내리려다 환경이 깨진 것이 지난번 실패의 큰 원인이었으니, 절대 다운그레이드하지 말 것.

---

## 셀 1 — 런타임 확인

먼저 메뉴에서 **런타임 → 런타임 유형 변경 → T4 GPU** 를 선택한다.

```python
!nvidia-smi -L
import torch; print(torch.__version__, torch.cuda.is_available())
```

## 셀 2 — 코드와 데이터 올리기

이 `재분석` 폴더를 구글 드라이브에 그대로 복사해 두고:

```python
from google.colab import drive
drive.mount('/content/drive')

PROJ = '/content/drive/MyDrive/재분석'     # 드라이브에 올린 폴더 경로
!cp -r "$PROJ" /content/work
%cd /content/work
!ls
```

`data_clean.csv`(1,431편)는 이미 만들어져 있으므로 `prepare_data.py`를 다시 돌릴 필요는 없다.
학습 결과(OOF)는 드라이브에 직접 저장해서 세션이 끊겨도 남게 한다.

```python
OOF = PROJ + '/oof'          # 결과 저장 위치 (드라이브)
```

## 셀 3 — 패키지 설치

Colab에는 torch·sklearn·pandas·numpy가 이미 있다. 없는 것만 넣는다.

```python
!pip -q install "transformers>=4.44" accelerate sentencepiece regex python-mecab-ko
```

`python-mecab-ko`는 pip 휠에 mecab-ko-dic이 들어 있어 **apt 빌드가 필요 없다.**
지난번에 쓴 `Mecab-ko-for-Google-Colab` 소스 빌드 스크립트는 원본 tarball 링크가
죽어서 요즘 자주 실패한다. 그 경로는 쓰지 않는다.

## 셀 4 — Mecab-ko 동작 확인 (KorSciDeBERTa 토크나이저용)

```python
!python mecab_compat.py
```

형태소 분석 결과가 출력되면 원본 토크나이저 경로로 학습된다.
실패해도 `finetune.py`가 같은 `spm.model` 기반 표준 토크나이저로 자동 강등해 학습은 진행된다
(그 경우 논문에 "사전학습과 동일한 토크나이저를 사용했다"고 쓰면 안 된다).

## 셀 5 — 먼저 1시드로 방향 확인 (모델당 약 20~35분)

```python
!python baseline_tfidf.py --seeds 42 --outdir "$OOF"
!python finetune.py --model klue/roberta-base  --tag klue-roberta  --seeds 42 --outdir "$OOF"
!python finetune.py --model kisti/korscideberta --tag korscideberta --seeds 42 --outdir "$OOF"
!python report.py --outdir "$OOF" --out "$PROJ/results/report_seed42.md"
```

여기서 두 모델의 순위와 TF-IDF(0.618) 대비 이득이 보인다.
이 단계에서 이미 논문의 핵심 표가 나온다.

## 셀 6 — 시드를 늘려 최종 수치 확정 (총 2~3시간)

```python
!python finetune.py --model klue/roberta-base  --tag klue-roberta  --seeds 42 43 44 --outdir "$OOF"
!python finetune.py --model kisti/korscideberta --tag korscideberta --seeds 42 43 44 --outdir "$OOF"
!python baseline_tfidf.py --seeds 42 43 44 --outdir "$OOF"
!python report.py --outdir "$OOF" --out "$PROJ/results/report.md"
```

이미 끝난 fold는 `oof/parts/`에 저장돼 있어 **건너뛴다.** 세션이 끊겨도 셀 2·3·6만 다시
실행하면 중단된 지점부터 이어진다.

---

## 문제가 생기면

| 증상 | 대처 |
|---|---|
| `CUDA out of memory` | `--bs 8 --grad-accum 2` (유효 배치는 16 유지) |
| 너무 느림 | `--max-len 256` (문서 95%가 448자 이내라 손실 작음) |
| 세션이 자꾸 끊김 | 셀 6을 그대로 재실행. 완료된 fold는 자동 스킵 |
| GPU 할당 거부(무료 한도) | 시드 1개로만 돌리고 다음 날 이어서 시드 추가 |
| `korscideberta.zip` 다운로드가 보임 | 구버전 코드다. `finetune.py`의 `allow_patterns` 확인 |

## 예상 소요 시간 (T4 기준)

- KLUE-RoBERTa-base: fold당 3~5분 → 1시드 20분, 3시드 약 1시간
- KorSciDeBERTa: disentangled attention 때문에 1.5~2배 느림 → 3시드 약 1.5~2시간

무료 Colab의 하루 GPU 사용량으로는 **1시드씩 이틀에 나눠 돌리는 편**이 안전하다.
`oof/parts/`가 드라이브에 남으므로 이어서 하면 된다.
