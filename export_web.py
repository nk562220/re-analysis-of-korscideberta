# -*- coding: utf-8 -*-
"""파인튜닝 모델 -> 브라우저에서 돌아가는 ONNX(int8)로 변환.

왜 ONNX인가:
  GitHub Pages 는 정적 파일만 서비스한다(파이썬 실행 불가).
  모델을 ONNX로 바꾸면 transformers.js 가 브라우저 안에서 직접 추론하므로
  서버도, API 키도, 비용도 없이 배포된다. 데이터도 밖으로 나가지 않는다.
  int8 양자화로 용량이 약 1/4(440MB -> 약 110MB)로 줄고 CPU 추론이 빨라진다.

사용법:
    pip install "optimum[onnxruntime]" onnx
    python export_web.py --model final_model --out web/model
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="final_model")
    ap.add_argument("--out", default="web/model")
    ap.add_argument("--no-quantize", action="store_true", help="양자화 생략(fp32, 용량 4배)")
    ap.add_argument("--probe", default="이산화티타늄 광촉매를 코팅한 섬유의 미세플라스틱 제거 효율을 측정하였다.")
    args = ap.parse_args()

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    tmp = os.path.join(args.out, "_fp32")
    os.makedirs(tmp, exist_ok=True)

    print("[1/4] ONNX 변환")
    ort_model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
    ort_model.save_pretrained(tmp)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(tmp)

    src = tmp
    if not args.no_quantize:
        print("[2/4] int8 양자화")
        try:
            qdir = os.path.join(args.out, "_int8")
            quantizer = ORTQuantizer.from_pretrained(tmp)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=qdir, quantization_config=qconfig)
            tok.save_pretrained(qdir)
            src = qdir
        except Exception as e:
            print("    양자화 실패(%s) -> fp32 로 진행" % e)
    else:
        print("[2/4] 양자화 생략")

    # transformers.js 가 기대하는 배치: 루트에 설정/토크나이저, onnx/ 아래에 가중치
    print("[3/4] transformers.js 배치로 정리")
    onnx_dir = os.path.join(args.out, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    weight = None
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if not os.path.isfile(p):
            continue
        if f.endswith(".onnx") or f.endswith(".onnx_data"):
            # 양자화 산출물 이름은 버전마다 다르다(model_quantized.onnx / model_int8.onnx 등)
            name = "model_quantized.onnx" if (not args.no_quantize and src.endswith("_int8")) else "model.onnx"
            if f.endswith(".onnx"):
                shutil.copy(p, os.path.join(onnx_dir, name))
                weight = name
            else:
                shutil.copy(p, os.path.join(onnx_dir, f))
        else:
            shutil.copy(p, os.path.join(args.out, f))

    for d in (tmp, os.path.join(args.out, "_int8")):
        shutil.rmtree(d, ignore_errors=True)

    print("[4/4] 검증: 같은 문장을 PyTorch/ONNX 로 각각 추론")
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification

    with open(os.path.join(args.out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}

    enc = tok(args.probe, return_tensors="pt", truncation=True, max_length=384)
    torch_model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    with torch.no_grad():
        p_torch = torch.softmax(torch_model(**enc).logits, -1)[0].numpy()
    onnx_reload = ORTModelForSequenceClassification.from_pretrained(
        args.out, file_name=os.path.join("onnx", weight)
    )
    p_onnx = torch.softmax(torch.tensor(onnx_reload(**enc).logits), -1)[0].numpy()

    print("    %-12s %-10s %-10s" % ("분야", "PyTorch", "ONNX"))
    for i in np.argsort(-p_torch):
        print("    %-12s %-10.4f %-10.4f" % (id2label.get(i, i), p_torch[i], p_onnx[i]))
    gap = float(np.abs(p_torch - p_onnx).max())
    print("    최대 확률 차이: %.4f %s" % (gap, "(정상)" if gap < 0.08 else "(※ 크다 — --no-quantize 로 재시도)"))

    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(args.out) for f in fs) / 1e6
    print("\n완료: %s  (총 %.0f MB)" % (os.path.abspath(args.out), size))
    print("""
다음 단계 — 모델 파일은 GitHub(100MB 제한)이 아니라 Hugging Face Hub에 둔다:

    pip install huggingface_hub
    huggingface-cli login
    huggingface-cli upload <내아이디>/korpaper-cls ./%s . --repo-type=model

그다음 web/index.html 을 열어 상단 입력란에 '<내아이디>/korpaper-cls' 를 넣으면 끝이다.
web/ 폴더만 GitHub 저장소에 올려 Pages 로 배포한다.
""" % args.out)


if __name__ == "__main__":
    main()
