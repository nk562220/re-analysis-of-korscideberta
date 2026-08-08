# 처음부터 끝까지 — Colab 실행 순서

빈 Colab 노트 하나에 아래 셀을 **위에서 아래로** 붙여넣고 실행한다.
노트북 파일(.ipynb)을 따로 만들 필요는 없다. 코드는 전부 이 저장소의 `.py` 파일이다.

**먼저 런타임 → 런타임 유형 변경 → T4 GPU → 저장.**

---

## 한 방에 실행 (셀 하나)

단계별로 보고 싶으면 아래 "셀 1~10" 절로 내려간다. 그냥 끝까지 돌리려면 이 블록 하나면 된다.
이미 끝난 fold는 `oof/parts/`에서 자동으로 건너뛴다.

```python
# ══ 0. 설정 ═══════════════════════════════════════════════════════
REPO  = 'https://github.com/nk562220/re-analysis-of-korscideberta.git'
HF_ID = 'nk562220/korpaper-cls'     # 업로드할 Hugging Face 모델 저장소
SEEDS = '42'                        # '42 43 44' 로 바꾸면 신뢰도↑ (시간 3배)

# ══ 1. 드라이브 연결 + 코드 받기 ══════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import glob, os
hits = glob.glob('/content/drive/MyDrive/**/data_clean.csv', recursive=True)
assert hits, 'data_clean.csv 를 드라이브에서 찾지 못했습니다. 업로드 위치를 확인하세요.'
DATA, PROJ = hits[0], os.path.dirname(hits[0])
OOF = PROJ + '/oof'
print('DATA =', DATA)

os.chdir('/content')
!rm -rf work && git clone -q {REPO} work
!cp "{DATA}" /content/work/
os.chdir('/content/work')

# ══ 2. 패키지 ═════════════════════════════════════════════════════
# 의존성 충돌 경고(gradio 등)는 무시해도 된다.
!pip -q install "transformers>=4.44" accelerate sentencepiece regex \
                python-mecab-ko "optimum[onnxruntime]" onnx
!nvidia-smi -L

# ══ 3. Hugging Face 로그인 ════════════════════════════════════════
# 지금 입력해 두면 이후 40분을 무인으로 돌릴 수 있다.
# 토큰: huggingface.co/settings/tokens 에서 write 권한으로 발급
from getpass import getpass
from huggingface_hub import login
login(token=getpass('HF write 토큰 붙여넣고 Enter: '))

# ══ 4. 성능 측정 — 논문에 쓸 수치 ═════════════════════════════════
!python baseline_tfidf.py --seeds 42 43 44 --outdir "{OOF}"
!python finetune.py --model klue/roberta-base   --tag klue-roberta   --seeds {SEEDS} --outdir "{OOF}"
!python mecab_compat.py
!python finetune.py --model kisti/korscideberta --tag korscideberta --seeds {SEEDS} --outdir "{OOF}"
!python report.py --outdir "{OOF}" --out "{PROJ}/results/report.md"

# ══ 5. 배포용 최종 모델 + ONNX 변환 ═══════════════════════════════
!python train_final.py --model klue/roberta-base --out final_model
!cp -r final_model "{PROJ}/"
!python export_web.py --model final_model --out docs/model

# ══ 6. 모델 업로드 ════════════════════════════════════════════════
!huggingface-cli upload {HF_ID} ./docs/model . --repo-type=model

print('\n' + '='*60)
print('완료. 모델: https://huggingface.co/' + HF_ID)
print('성능 표: ' + PROJ + '/results/report.md')
print("다음: docs/index.html 의 DEFAULT_MODEL 을 '" + HF_ID + "' 로 변경")
print('='*60)
```

> ⚠️ 파이썬 버전을 절대 내리지 말 것.
> KISTI 문서의 "Python 3.8~3.9 / torch 1.10" 제약은 원본 DeBERTa 학습 저장소용이며,
> `transformers`로 모델을 쓰는 이 코드에는 해당되지 않는다.
> 지난 연구에서 이 오해로 이틀을 소모했다.

---

## 셀 1 — 코드 받기 + 데이터 연결

코드는 **GitHub에서** 받고, 데이터(`data_clean.csv`)는 **드라이브에서** 가져온다.
데이터는 학생 저작물이라 공개 저장소에 넣지 않았다.

```python
from google.colab import drive
drive.mount('/content/drive')

import glob, os
DATA = glob.glob('/content/drive/MyDrive/**/data_clean.csv', recursive=True)[0]
PROJ = os.path.dirname(DATA)          # 드라이브의 프로젝트 폴더
OOF  = PROJ + '/oof'                  # 학습 결과 저장 위치(끊겨도 남음)
print('DATA =', DATA)

%cd /content
!rm -rf work
!git clone -q https://github.com/nk562220/re-analysis-of-korscideberta.git work
!cp "{DATA}" /content/work/
%cd /content/work
!ls
```

`data_clean.csv`와 `.py` 파일들이 보이면 정상이다.
**코드가 갱신될 때마다 이 셀만 다시 실행하면 된다**(`git pull`이나 `git stash`를 쓰지 말고
새로 clone하는 편이 안전하다 — stash는 데이터 파일까지 치워버린다).

## 셀 2 — 패키지 설치

Colab에는 torch·sklearn·pandas가 이미 있다. 없는 것만 넣는다.

```python
!pip -q install "transformers>=4.44" accelerate sentencepiece regex python-mecab-ko
!nvidia-smi -L
```

`Tesla T4` 같은 GPU 이름이 나와야 한다. 안 나오면 런타임 유형을 다시 확인한다.

`gradio ... requires huggingface-hub>=1.2.0` 류의 의존성 경고는 무시한다.
Colab에 미리 깔린 다른 패키지와의 충돌 알림이고, 이 파이프라인에는 영향이 없다.

---

# A. 논문에 쓸 성능 수치 구하기

## 셀 3 — 고전 베이스라인 (30초)

사전학습 모델이 의미 있는 이득을 주는지 판단할 기준선이다. 이것 없이는 성능 평가가 성립하지 않는다.

```python
!python baseline_tfidf.py --seeds 42 43 44 --outdir "{OOF}"
```

## 셀 4 — 범용 한국어 모델 (시드당 약 12분)

```python
!python finetune.py --model klue/roberta-base --tag klue-roberta --seeds 42 --outdir "{OOF}"
```

## 셀 5 — 과학기술 특화 모델 (시드당 약 20분)

KorSciDeBERTa 원본 토크나이저(Mecab-ko)를 쓸 수 있는지 먼저 확인한다.
실패해도 같은 `spm.model` 기반 표준 토크나이저로 자동 강등되어 학습은 진행된다.

```python
!python mecab_compat.py
!python finetune.py --model kisti/korscideberta --tag korscideberta --seeds 42 --outdir "{OOF}"
```

## 셀 6 — 결과 표 만들기

```python
!python report.py --outdir "{OOF}" --out "{PROJ}/results/report.md"
```

정확도·macro-F1·95% 신뢰구간·혼동행렬·모델 간 유의성 검정이 한 번에 나온다.
**여기서 나온 수치가 논문에 쓸 수치다.**

시드를 늘리면 `± 표준편차`가 채워져 신뢰도가 올라간다(이미 끝난 fold는 자동으로 건너뛴다):

```python
!python finetune.py --model klue/roberta-base   --tag klue-roberta   --seeds 42 43 44 --outdir "{OOF}"
!python finetune.py --model kisti/korscideberta --tag korscideberta --seeds 42 43 44 --outdir "{OOF}"
!python report.py --outdir "{OOF}" --out "{PROJ}/results/report.md"
```

---

# B. 웹 분류기 배포

## 셀 7 — 배포용 최종 모델 학습 (약 3분)

A단계의 fold 모델은 데이터의 80%만 보고 학습 후 삭제된다.
실제로 쓸 모델은 전체 데이터로 다시 학습한다.

```python
!python train_final.py --model klue/roberta-base --out final_model
!cp -r final_model "{PROJ}/"          # 런타임이 끊겨도 남게 드라이브에 백업
```

여기 출력되는 검증 정확도는 조기종료용이므로 **논문에 인용하지 않는다.**

## 셀 8 — ONNX 변환 (약 2분)

GitHub Pages는 파이썬을 못 돌린다. ONNX로 바꾸면 브라우저가 직접 추론하므로
서버·API 키·비용이 전부 필요 없고 입력한 초록도 외부로 나가지 않는다.

```python
!pip -q install "optimum[onnxruntime]" onnx
!python export_web.py --model final_model --out docs/model
```

PyTorch와 ONNX의 확률이 나란히 출력된다. **최대 차이가 0.08 미만이면 정상.**
크게 벌어지면 `--no-quantize`로 다시 만든다(용량 4배, 정확도 손실 없음).

## 셀 9 — Hugging Face Hub에 모델 올리기

GitHub은 파일당 100MB 제한이라 모델이 안 들어간다. HF Hub는 용량이 넉넉하고
CORS가 열려 있어 브라우저가 바로 받아갈 수 있다.

```python
!huggingface-cli login
```

[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)에서 **write 권한** 토큰을 발급해 붙여넣는다.

```python
!huggingface-cli upload nk562220/korpaper-cls ./docs/model . --repo-type=model
```

## 셀 10 — 마무리

`docs/index.html`의 `DEFAULT_MODEL`을 방금 만든 모델 ID로 바꿔 커밋·push한다.

```js
const DEFAULT_MODEL = 'nk562220/korpaper-cls';
```

그리고 저장소 **Settings → Pages → Source: `Deploy from a branch` → Branch: `main` / `/docs` → Save**.

1~2분 뒤 https://nk562220.github.io/re-analysis-of-korscideberta/ 에서 열린다.

---

## 문제가 생기면

| 증상 | 원인과 대처 |
|---|---|
| `can't open file '...py'` | 셀 1을 다시 실행(코드가 갱신됨) |
| `CUDA out of memory` | 명령 끝에 `--bs 8 --grad-accum 2` |
| 너무 느림 | `--max-len 256` (문서 95%가 448자 이내) |
| 세션이 끊김 | 셀 1·2 후 끊긴 셀만 재실행. 끝난 fold는 `oof/parts/`에서 자동 스킵 |
| `final_model`이 사라짐 | 셀 7 재실행(3분) 또는 드라이브 백업에서 복사 |
| GPU 할당 거부 | 시드 1개만 돌리고 다음 날 이어서 추가 |

## 소요 시간 요약 (T4)

| 단계 | 시간 |
|---|---|
| 셀 3 TF-IDF 3시드 | 30초 |
| 셀 4 KLUE-RoBERTa 1시드 | 약 12분 |
| 셀 5 KorSciDeBERTa 1시드 | 약 20분 |
| 셀 7 최종 모델 | 약 3분 |
| 셀 8 ONNX 변환 | 약 2분 |
