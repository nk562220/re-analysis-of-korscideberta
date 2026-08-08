# 분류기 웹 UI 배포

`index.html` 하나로 끝난다. **서버가 필요 없다.**
ONNX로 변환한 모델을 transformers.js가 브라우저 안에서 직접 실행하므로
GitHub Pages(정적 호스팅)에 그대로 올릴 수 있고, 비용도 API 키도 없다.
입력한 논문 내용이 외부로 전송되지 않는다는 것도 장점이다.

```
사용자 브라우저
 ├── index.html          ← GitHub Pages
 └── ONNX 모델 (110MB)   ← Hugging Face Hub (최초 1회 다운로드 후 캐시)
```

## 왜 KLUE-RoBERTa를 배포하는가

교차검증에서 KLUE-RoBERTa(0.673)가 KorSciDeBERTa(0.623)보다 높았고,
**KorSciDeBERTa는 애초에 브라우저에서 돌릴 수 없다.** 그 토크나이저는
Mecab-ko 형태소 분석기(C++ 바이너리 + 사전)를 요구하는데 이걸 브라우저에
넣을 방법이 없다. KLUE-RoBERTa는 vocab.txt 기반 WordPiece라 문제없다.

## 1. 최종 모델 학습 (Colab, 약 3분)

교차검증용 fold 모델은 데이터의 80%만 봤고 이미 삭제되었다. 전체로 다시 학습한다.

```python
!python train_final.py --model klue/roberta-base --out final_model
```

## 2. ONNX 변환 (Colab, 약 3분)

```python
!pip -q install "optimum[onnxruntime]" onnx
!python export_web.py --model final_model --out docs/model
```

PyTorch와 ONNX의 출력 확률을 비교해서 찍어준다. **최대 차이가 0.08 미만이면 정상**이다.
크게 벌어지면 `--no-quantize`로 다시 만든다(용량 4배, 정확도 손실 없음).

## 3. 모델을 Hugging Face Hub에 올리기

GitHub는 파일 하나당 100MB 제한이라 int8 모델(약 110MB)이 안 들어간다.
HF Hub는 용량 제한이 넉넉하고 CORS도 열려 있어 브라우저가 바로 받아갈 수 있다.

```python
!pip -q install huggingface_hub
!huggingface-cli login          # HF 토큰 입력 (Settings → Access Tokens에서 write 권한으로 발급)
!huggingface-cli upload <내아이디>/korpaper-cls ./docs/model . --repo-type=model
```

## 4. GitHub Pages 배포

이 폴더는 이미 저장소에 포함되어 있다. 모델 파일(`docs/model/`)은 `.gitignore`로
제외되어 있으므로 올라가는 것은 `index.html`과 이 README뿐이다.

GitHub Pages는 브랜치 배포 시 `/ (root)`와 `/docs` 두 곳만 선택할 수 있다.
그래서 이 폴더 이름이 `web`이 아니라 `docs`다.

저장소 **Settings → Pages → Source: `Deploy from a branch` → Branch: `main` / `/docs` → Save**.
1~2분 뒤 https://nk562220.github.io/re-analysis-of-korscideberta/ 에서 열린다.

`index.html`의 `DEFAULT_MODEL`을 자기 HF 모델 ID로 바꿔 두면 방문자가 설정 없이 바로 쓸 수 있다.

```js
const DEFAULT_MODEL = '<내아이디>/korpaper-cls';
```

## 로컬에서 먼저 확인하기

`file://`로 직접 열면 브라우저 보안 정책(CORS) 때문에 모델을 못 읽는다.
반드시 간단한 서버를 띄워서 확인한다.

```bash
cd web
python -m http.server 8000
```

`http://localhost:8000` 접속 → 모델 설정에 `model` 입력 → 다시 불러오기.
(`docs/model/` 폴더가 있어야 한다.)

## 문제 해결

| 증상 | 원인과 대처 |
|---|---|
| `모델을 불러오지 못했습니다: 404` | 모델 ID 오타, 또는 HF 저장소가 private. public으로 바꾼다 |
| 로컬에서만 실패 | `file://`로 열었을 가능성. `python -m http.server` 사용 |
| 첫 로딩이 아주 느림 | 정상. 110MB를 받는다. 이후에는 브라우저 캐시에서 즉시 로드 |
| 결과가 전부 비슷한 확률 | 입력이 너무 짧다. 초록 2~3문장 이상 넣는다 |
| 확률이 학습 때와 다름 | 제목·초록을 따로 넣었는지 확인. UI가 `제목 [SEP] 초록`으로 합쳐야 맞다 |
