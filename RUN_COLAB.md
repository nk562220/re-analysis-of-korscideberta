# 처음부터 끝까지 — Colab 실행 순서

빈 Colab 노트 하나에 아래 셀을 **위에서 아래로** 붙여넣고 실행한다.
노트북 파일(.ipynb)을 따로 만들 필요는 없다. 코드는 전부 이 저장소의 `.py` 파일이다.

**먼저 런타임 → 런타임 유형 변경 → T4 GPU → 저장.**

---

## 한 방에 실행 (셀 두 개)

단계별로 보고 싶으면 아래 "셀 1~10" 절로 내려간다. 그냥 끝까지 돌리려면 이 두 블록이면 된다.
이미 끝난 fold는 `oof/parts/`에서 자동으로 건너뛴다.

### 셀 A — 패키지 설치

`optimum` 설치는 Colab에 미리 깔린 `huggingface_hub`을 다른 버전으로 교체하면서
파일이 섞이고 `ImportError: cannot import name 'XetAuthorizationError'` 를 일으킨다.
`--force-reinstall` 로 그 패키지만 깨끗하게 덮어써서 해결한다.

`os.kill` 로 커널을 강제 종료하는 방식은 쓰지 않는다. Colab이 이를 세션 충돌로 처리해
VM을 새로 잡아버리면 방금 설치한 패키지까지 사라져 무한 반복이 된다.
셀 B의 학습·변환은 모두 `!python 스크립트` 즉 **별도 프로세스**에서 실행되므로
세션 안에서 `huggingface_hub` 을 임포트하지 않는 한 재시작은 필요 없다.

```python
!pip -q install "transformers>=4.44" accelerate sentencepiece regex \
                python-mecab-ko "optimum[onnxruntime]" onnx
!pip -q install --force-reinstall --no-deps "huggingface_hub==0.36.2"

# 새 인터프리터에서 임포트해 설치가 온전한지 확인한다(세션은 건드리지 않는다)
!python -c "import huggingface_hub, transformers, optimum, onnxruntime as o; \
print('설치 정상:', huggingface_hub.__version__, transformers.__version__, o.__version__)"
```

마지막 줄에 `설치 정상: ...` 이 찍히면 셀 B로 넘어간다.
`ImportError` 가 찍히면 그때만 메뉴에서 **런타임 → 세션 다시 시작** 후 셀 B를 실행한다.

### 셀 B — 전체 파이프라인

```python
# ══ 0. 설정 ═══════════════════════════════════════════════════════
REPO  = 'https://github.com/nk562220/re-analysis-of-korscideberta.git'
HF_ID = 'nk56/korpaper-cls'     # 업로드할 Hugging Face 모델 저장소
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
!nvidia-smi -L

# ══ 2. Hugging Face 토큰 ══════════════════════════════════════════
# huggingface_hub 을 세션에서 임포트하지 않고 환경변수로 넘긴다.
# huggingface-cli 가 HF_TOKEN 을 자동으로 읽으므로 login() 이 필요 없고,
# 버전 충돌과 런타임 재시작을 모두 피할 수 있다.
# 토큰: huggingface.co/settings/tokens 에서 write 권한으로 발급
from getpass import getpass
os.environ['HF_TOKEN'] = getpass('HF write 토큰 붙여넣고 Enter: ').strip()

# ══ 3. 성능 측정 — 논문에 쓸 수치 ═════════════════════════════════
!python baseline_tfidf.py --seeds 42 43 44 --outdir "{OOF}"
!python finetune.py --model klue/roberta-base   --tag klue-roberta   --seeds {SEEDS} --outdir "{OOF}"
!python mecab_compat.py
!python finetune.py --model kisti/korscideberta --tag korscideberta --seeds {SEEDS} --outdir "{OOF}"
!python report.py --outdir "{OOF}" --out "{PROJ}/results/report.md"

# ══ 4. 배포용 최종 모델 + ONNX 변환 ═══════════════════════════════
!python train_final.py --model klue/roberta-base --out final_model
!cp -r final_model "{PROJ}/"
!python export_web.py --model final_model --out docs/model

# ══ 5. 모델 업로드 ════════════════════════════════════════════════
# 종료 코드를 검사한다. !명령 은 실패해도 셀이 계속 진행되므로
# 검사하지 않으면 업로드가 실패했는데 '완료'가 찍힌다.
rc = os.system(f'hf upload {HF_ID} ./docs/model . --repo-type=model')
assert rc == 0, '업로드 실패 — 위 오류 메시지를 확인하세요(토큰은 hf_ 로 시작하는 37자)'

print('\n' + '='*60)
print('완료. 모델: https://huggingface.co/' + HF_ID)
print('성능 표: ' + PROJ + '/results/report.md')
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

[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)에서 **write 권한** 토큰을 발급해
아래에 붙여넣는다. 화면에 찍히지 않고 노트북에도 저장되지 않는다.

```python
import os
from getpass import getpass
os.environ['HF_TOKEN'] = getpass('HF write 토큰 붙여넣고 Enter: ').strip()
assert os.environ['HF_TOKEN'].startswith('hf_'), '토큰 형식이 아닙니다. 발급 팝업의 복사 버튼으로 다시 복사하세요.'

rc = os.system('hf upload nk56/korpaper-cls ./docs/model . --repo-type=model')
assert rc == 0, '업로드 실패 — 위 오류 확인'
```

`hf` 는 `HF_TOKEN` 환경변수를 자동으로 읽으므로 `login` 을 따로 하지 않는다.
세션에서 `huggingface_hub` 을 임포트하지 않아 버전 충돌도 피할 수 있다.

토큰은 반드시 **발급 직후 팝업의 복사 버튼**으로 복사한다(`hf_` + 34자).
목록 화면에는 이름만 보이므로 거기서 복사하면 토큰이 아닌 값이 잡혀 401이 난다.
`huggingface-cli` 는 폐기 예고가 떴으므로 새 명령 `hf` 를 쓴다.

## 셀 10 — 마무리

`docs/index.html` 의 `DEFAULT_MODEL` 이 이미 아래 값으로 설정되어 있다(변경 완료).

```js
const DEFAULT_MODEL = 'nk56/korpaper-cls';
```

그리고 저장소 **Settings → Pages → Source: `Deploy from a branch` → Branch: `main` / `/docs` → Save**.

1~2분 뒤 https://nk562220.github.io/re-analysis-of-korscideberta/ 에서 열린다.

---

## 문제가 생기면

| 증상 | 원인과 대처 |
|---|---|
| `can't open file '...py'` | 셀 1을 다시 실행(코드가 갱신됨) |
| `ImportError: cannot import name 'XetAuthorizationError'` | pip 설치 후 런타임을 재시작하지 않은 것. 셀 A를 실행해 재시작한 뒤 셀 B를 돌린다 |
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
