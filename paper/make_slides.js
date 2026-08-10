// 발표 자료(PPTX) 생성 — 5분 발표 기준 10장
//
// 실행:
//   NODE_PATH=<node_modules 경로> node paper/make_slides.js
//
// 색: 화이트 배경 / 제목 네이비 1B365D / 포인트 오렌지 E8833A
// 수치는 results/report.md 의 실측값(시드 42 OOF, 1,431편)

const pptxgen = require("pptxgenjs");
const path = require("path");

const NAVY = "1B365D";
const ORANGE = "E8833A";
const GRAY = "5A6472";
const TINT = "F2F5F9";
const WHITE = "FFFFFF";
const RED = "C0392B";
const KR = "Malgun Gothic";

const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "청소년논문분류_재검증_발표.pptx");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 inch — 슬라이드 추가 전에 지정해야 한다
pres.author = "2220 최은수B, 2105 김경민";
pres.title = "청소년 과학 논문의 분야 자동 분류 재검증";

// 카드용 그림자. pptxgenjs 는 옵션 객체를 그 자리에서 변형하므로 호출마다 새로 만든다.
const shadow = () => ({ type: "outer", color: "9AA5B4", blur: 8, offset: 2, angle: 90, opacity: 0.25 });

function title(slide, text, sub) {
  slide.addText(text, {
    x: 0.6, y: 0.42, w: 12.1, h: 0.7, fontFace: KR, fontSize: 32, bold: true,
    color: NAVY, margin: 0,
  });
  if (sub) {
    slide.addText(sub, {
      x: 0.6, y: 1.14, w: 12.1, h: 0.4, fontFace: KR, fontSize: 15,
      color: GRAY, margin: 0,
    });
  }
}

// 번호 원 + 제목 + 설명 카드 (반복되는 시각 모티프)
function numberCard(slide, n, y, head, body, accent) {
  slide.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: y, w: 12.1, h: 1.42, fill: { color: TINT }, rectRadius: 0.08,
    line: { color: TINT }, shadow: shadow(),
  });
  slide.addShape(pres.ShapeType.ellipse, {
    x: 0.95, y: y + 0.36, w: 0.7, h: 0.7, fill: { color: accent ? ORANGE : NAVY },
    line: { color: accent ? ORANGE : NAVY },
  });
  slide.addText(String(n), {
    x: 0.95, y: y + 0.36, w: 0.7, h: 0.7, fontFace: KR, fontSize: 22, bold: true,
    color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  slide.addText(head, {
    x: 1.85, y: y + 0.24, w: 10.5, h: 0.42, fontFace: KR, fontSize: 20, bold: true,
    color: NAVY, margin: 0, valign: "middle",
  });
  slide.addText(body, {
    x: 1.85, y: y + 0.7, w: 10.5, h: 0.55, fontFace: KR, fontSize: 15,
    color: GRAY, margin: 0, valign: "top",
  });
}

// ───────────────────────────────────────── 1. 표지
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("청소년 과학 논문의\n분야 자동 분류 재검증", {
    x: 1.0, y: 1.75, w: 11.3, h: 1.9, fontFace: KR, fontSize: 40, bold: true,
    color: WHITE, lineSpacing: 46, margin: 0,
  });
  s.addText("평가 설계 오류의 교정과 사전학습 언어모델 비교", {
    x: 1.0, y: 3.85, w: 11.3, h: 0.5, fontFace: KR, fontSize: 20,
    color: ORANGE, margin: 0,
  });
  s.addText("연구자  2220 최은수B · 2105 김경민      지도교사  한정신", {
    x: 1.0, y: 5.6, w: 11.3, h: 0.4, fontFace: KR, fontSize: 15,
    color: "AEBACC", margin: 0,
  });
  s.addNotes("선행 연구에서 정확도 0.0812를 보고했는데, 이 수치가 잘못된 측정이었음을 발견해 평가를 다시 설계한 연구입니다.");
}

// ───────────────────────────────────────── 2. 문제 발견
{
  const s = pres.addSlide();
  title(s, "왜 다시 했는가", "선행 연구가 보고한 정확도를 기준선과 나란히 놓아 보았다");

  const stats = [
    { v: "0.0812", l: "선행 연구 보고값", c: RED },
    { v: "0.2000", l: "무작위 추측 (5분류)", c: NAVY },
    { v: "0.2502", l: "최빈 클래스만 예측", c: NAVY },
  ];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 4.13;
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 1.95, w: 3.84, h: 2.3, fill: { color: TINT }, rectRadius: 0.08,
      line: { color: TINT }, shadow: shadow(),
    });
    s.addText(st.v, {
      x: x, y: 2.2, w: 3.84, h: 1.0, fontFace: KR, fontSize: 46, bold: true,
      color: st.c, align: "center", margin: 0,
    });
    s.addText(st.l, {
      x: x, y: 3.3, w: 3.84, h: 0.5, fontFace: KR, fontSize: 15,
      color: GRAY, align: "center", margin: 0,
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 4.75, w: 12.1, h: 1.35, fill: { color: NAVY }, rectRadius: 0.08,
    line: { color: NAVY },
  });
  s.addText("아무것도 학습하지 않은 모델보다 나쁜 성능은\n모델의 한계가 아니라 측정의 오류를 뜻한다", {
    x: 0.9, y: 4.9, w: 11.5, h: 1.05, fontFace: KR, fontSize: 21, bold: true,
    color: WHITE, align: "center", lineSpacing: 30, margin: 0,
  });
  s.addNotes("5분류 문제에서 무작위로 찍으면 0.2, 가장 많은 분야만 답해도 0.25입니다. 0.0812는 그보다 낮습니다. 여기서 측정 자체를 의심하게 됐습니다.");
}

// ───────────────────────────────────────── 3. 원인
{
  const s = pres.addSlide();
  title(s, "원인은 데이터가 아니라 코드였다", "선행 연구의 평가 코드를 다시 읽어 세 가지를 확인했다");
  numberCard(s, 1, 1.75, "학습 루프가 없었다",
    "AutoModelForSequenceClassification 은 분류 헤드를 무작위로 새로 만든다. 옵티마이저와 역전파가 없어 그 난수가 끝까지 유지됐다.", true);
  numberCard(s, 2, 3.35, "레이블 수가 맞지 않았다",
    "num_labels=7 로 설정했으나 정답은 클러스터 번호 0~2뿐이었다. 존재할 수 없는 4~6번이 예측 후보에 있었다.");
  numberCard(s, 3, 4.95, "평가가 순환 구조였다",
    "정답 레이블을 같은 임베딩의 클러스터링으로 만들었다. 논문 주제가 아니라 K-Means 의 결정을 얼마나 모방하는지를 재고 있었다.");
  s.addNotes("가장 큰 문제는 첫 번째입니다. 모델을 불러오면 경고가 출력되는데 그것을 놓쳤습니다. 두 번째와 세 번째도 각각 독립적으로 평가를 무너뜨립니다.");
}

// ───────────────────────────────────────── 4. 교정
{
  const s = pres.addSlide();
  title(s, "무엇을 바로잡았는가", "정확도를 높이기보다 먼저 믿을 수 있게 만들었다");

  const cards = [
    { h: "외부 정답 레이블", n: "5개 분야", b: "전국과학전람회가 부여한 공식 분야를 정답으로 삼았다. 우리 모델과 완전히 독립적이다." },
    { h: "데이터 누수 차단", n: "773 / 876", b: "지도논문을 학생논문에 병합(88%)하고, 유사 중복 문서는 같은 겹에 묶었다." },
    { h: "교차검증 OOF", n: "1,431편", b: "전체 문서가 각각 한 번씩, 자신을 학습하지 않은 모델에게 평가받는다." },
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 4.13;
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 1.8, w: 3.84, h: 3.6, fill: { color: TINT }, rectRadius: 0.08,
      line: { color: TINT }, shadow: shadow(),
    });
    s.addText(c.h, {
      x: x + 0.28, y: 2.05, w: 3.3, h: 0.45, fontFace: KR, fontSize: 19, bold: true,
      color: NAVY, margin: 0,
    });
    s.addText(c.n, {
      x: x + 0.28, y: 2.6, w: 3.3, h: 0.75, fontFace: KR, fontSize: 32, bold: true,
      color: ORANGE, margin: 0,
    });
    s.addText(c.b, {
      x: x + 0.28, y: 3.5, w: 3.3, h: 1.6, fontFace: KR, fontSize: 14,
      color: GRAY, margin: 0, valign: "top",
    });
  });
  s.addText("단일 홀드아웃(215편)보다 평가 표본이 6.6배 커져 수치가 훨씬 덜 흔들린다", {
    x: 0.6, y: 5.75, w: 12.1, h: 0.45, fontFace: KR, fontSize: 16, bold: true,
    color: NAVY, align: "center", margin: 0,
  });
  s.addNotes("세 가지가 모두 필요합니다. 외부 정답이 없으면 평가가 성립하지 않고, 누수를 막지 않으면 수치가 부풀려지고, 단일 분할이면 운에 좌우됩니다.");
}

// ───────────────────────────────────────── 5. 데이터
{
  const s = pres.addSlide();
  title(s, "데이터: 2,349편 → 1,431편", "전국과학전람회 2020–2024 수상 논문 (제목 + 요약문)");

  const rows = [
    [{ text: "처리 단계", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "편수", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } }],
    ["원본 수집", "2,349"],
    ["결측·완전중복 제거", "2,337"],
    ["지도논문 773편 병합", "1,564"],
    ["크롤링 실패 42편 제거", "1,522"],
    ["'기타' 분야 91편 제외", "1,431"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.85, w: 5.6, colW: [3.9, 1.7], rowH: 0.46,
    fontFace: KR, fontSize: 14, color: NAVY, border: { type: "solid", color: "D8DEE8", pt: 1 },
    align: "left", valign: "middle",
  });

  s.addChart(pres.ChartType.bar, [{
    name: "편수",
    labels: ["생물", "물리", "화학", "지구및환경", "산업및에너지"],
    values: [358, 327, 300, 290, 156],
  }], {
    x: 6.5, y: 1.8, w: 6.2, h: 3.7,
    barDir: "col", chartColors: [NAVY],
    showTitle: true, title: "분야별 편수", titleFontFace: KR, titleFontSize: 14, titleColor: NAVY,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: KR,
    dataLabelFontSize: 11, dataLabelColor: NAVY,
    showLegend: false, catAxisLabelFontFace: KR, catAxisLabelFontSize: 11,
    catAxisLabelColor: GRAY, valAxisLabelColor: GRAY, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" },
  });

  s.addText("요약문은 평균 254자로 짧다. 선행 연구가 쓴 512토큰은 과했고 384토큰으로 충분했다.", {
    x: 0.6, y: 5.75, w: 12.1, h: 0.45, fontFace: KR, fontSize: 15,
    color: GRAY, margin: 0,
  });
  s.addNotes("지도교사가 쓴 지도논문 876편이 학생논문과 내용이 거의 같아서, 나뉘어 들어가면 정확도가 부풀려집니다. 88%를 병합했습니다.");
}

// ───────────────────────────────────────── 6. 결과
{
  const s = pres.addSlide();
  title(s, "결과: 기준선을 6.8%p 상회", "1,431편 out-of-fold, 시드 42");
  s.addImage({ path: path.join(FIG, "fig1_scores.png"), x: 1.95, y: 1.62, w: 9.4, h: 4.7 });
  s.addText("TF-IDF 0.6054 → KLUE-RoBERTa 0.6730 (macro-F1 0.5896 → 0.6625)", {
    x: 0.6, y: 6.42, w: 12.1, h: 0.42, fontFace: KR, fontSize: 16, bold: true,
    color: NAVY, align: "center", margin: 0,
  });
  s.addNotes("기준선을 먼저 만든 이유가 여기 있습니다. 0.673이라는 숫자만 있으면 좋은지 나쁜지 알 수 없습니다. 단어 빈도만 쓰는 모델이 0.605니까, 사전학습의 기여는 6.8%p입니다.");
}

// ───────────────────────────────────────── 7. 예상 밖의 결과
{
  const s = pres.addSlide();
  title(s, "특화 모델이 범용 모델보다 낮았다", "선행 연구가 KorSciDeBERTa 를 선택한 근거와 반대되는 결과");

  const rows = [
    [{ text: "모델", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "Accuracy", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } },
     { text: "macro-F1", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "right" } }],
    [{ text: "KLUE-RoBERTa (범용, 110M)", options: { bold: true } },
     { text: "0.6730", options: { align: "right", bold: true } },
     { text: "0.6625", options: { align: "right", bold: true } }],
    ["KorSciDeBERTa (특화, 180M)", { text: "0.6233", options: { align: "right" } },
     { text: "0.6104", options: { align: "right" } }],
    [{ text: "차이", options: { color: ORANGE, bold: true } },
     { text: "+0.0497", options: { align: "right", color: ORANGE, bold: true } },
     { text: "+0.0521", options: { align: "right", color: ORANGE, bold: true } }],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.85, w: 6.5, colW: [3.3, 1.6, 1.6], rowH: 0.5,
    fontFace: KR, fontSize: 14, color: NAVY,
    border: { type: "solid", color: "D8DEE8", pt: 1 }, valign: "middle",
  });
  s.addText("짝지은 부트스트랩 95% CI [+0.028, +0.076], p < 0.001\n→ 표본 크기에 의한 우연으로 보기 어렵다", {
    x: 0.6, y: 4.15, w: 6.5, h: 0.9, fontFace: KR, fontSize: 14,
    color: GRAY, margin: 0, lineSpacing: 22,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 7.45, y: 1.85, w: 5.25, h: 3.9, fill: { color: TINT }, rectRadius: 0.08,
    line: { color: TINT }, shadow: shadow(),
  });
  s.addText("가능한 해석", {
    x: 7.75, y: 2.1, w: 4.65, h: 0.4, fontFace: KR, fontSize: 18, bold: true,
    color: NAVY, margin: 0,
  });
  s.addText([
    { text: "문체 차이 — 특화 모델은 논문·특허로 학습됐지만 청소년 논문은 짧고 구어적이다", options: { bullet: true, breakLine: true } },
    { text: "모델 크기 — 1.6배 큰 모델이 1,431편에서 과적합했을 수 있다", options: { bullet: true, breakLine: true } },
    { text: "학습률 — 두 모델에 같은 2e-5 를 적용했다. DeBERTa 는 학습률에 민감하다", options: { bullet: true } },
  ], {
    x: 7.75, y: 2.6, w: 4.65, h: 2.9, fontFace: KR, fontSize: 14, color: GRAY,
    paraSpaceAfter: 10, margin: 0, valign: "top",
  });

  s.addText("결론은 \"부적합하다\"가 아니라 \"같은 조건에서 도메인 이점이 전이되지 않았다\"로 한정한다", {
    x: 0.6, y: 6.0, w: 12.1, h: 0.45, fontFace: KR, fontSize: 15, bold: true,
    color: NAVY, align: "center", margin: 0,
  });
  s.addNotes("학습률을 바꾼 실험은 아직 하지 않았습니다. 그래서 결론을 조건부로 씁니다. 이것이 후속 과제입니다.");
}

// ───────────────────────────────────────── 8. 어디서 틀리나
{
  const s = pres.addSlide();
  title(s, "오분류의 상당수는 사람도 애매하다", "KLUE-RoBERTa 혼동행렬 (행=실제, 열=예측)");
  s.addImage({ path: path.join(FIG, "fig2_confusion.png"), x: 0.6, y: 1.6, w: 5.6, h: 4.75 });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.55, y: 1.72, w: 6.15, h: 2.5, fill: { color: TINT }, rectRadius: 0.08,
    line: { color: TINT }, shadow: shadow(),
  });
  s.addText("확신도가 높았던 오답", {
    x: 6.85, y: 1.92, w: 5.55, h: 0.38, fontFace: KR, fontSize: 17, bold: true,
    color: NAVY, margin: 0,
  });
  s.addText([
    { text: "「고수 추출물을 활용한 항균 마스크」\n정답 화학 → 예측 생물 (확신도 0.94)", options: { breakLine: true } },
    { text: "「커피박을 이용한 토양 미세플라스틱 제거」\n정답 생물 → 예측 지구및환경 (확신도 0.93)", options: {} },
  ], {
    x: 6.85, y: 2.38, w: 5.55, h: 1.7, fontFace: KR, fontSize: 13.5, color: GRAY,
    paraSpaceAfter: 12, lineSpacing: 19, margin: 0, valign: "top",
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.55, y: 4.42, w: 6.15, h: 1.93, fill: { color: NAVY }, rectRadius: 0.08,
    line: { color: NAVY },
  });
  s.addText("0.8679", {
    x: 6.85, y: 4.62, w: 5.55, h: 0.85, fontFace: KR, fontSize: 40, bold: true,
    color: ORANGE, margin: 0,
  });
  s.addText("Top-2 정확도 — 상위 두 분야 안에 정답이 있는 비율.\n정확도 100%는 도달 가능한 목표가 아니다.", {
    x: 6.85, y: 5.5, w: 5.55, h: 0.7, fontFace: KR, fontSize: 13.5,
    color: "D6DEEA", margin: 0, lineSpacing: 19,
  });
  s.addNotes("항균 마스크는 화학과 생물 어느 쪽으로도 분류될 수 있습니다. 성능의 상한은 심사 분야 배정 자체의 일관성이 정합니다.");
}

// ───────────────────────────────────────── 9. 왜 교차검증인가
{
  const s = pres.addSlide();
  title(s, "단일 분할로 재면 7%p까지 흔들린다", "같은 모델·같은 데이터에서 겹만 바꿨을 때의 정확도");
  s.addImage({ path: path.join(FIG, "fig3_folds.png"), x: 2.35, y: 1.72, w: 8.6, h: 4.3 });
  s.addText("겹 3만 떼어 보고하면 0.715, 겹 4만 보면 0.645. 교차검증과 신뢰구간을 함께 써야 한다.", {
    x: 0.6, y: 6.2, w: 12.1, h: 0.45, fontFace: KR, fontSize: 16, bold: true,
    color: NAVY, align: "center", margin: 0,
  });
  s.addNotes("선행 연구가 단일 실행으로 수치를 보고한 것도 문제였습니다. 이 그래프가 그 위험을 실측으로 보여줍니다.");
}

// ───────────────────────────────────────── 10. 배포 + 교훈
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("배운 것", {
    x: 0.9, y: 0.6, w: 11.5, h: 0.7, fontFace: KR, fontSize: 32, bold: true,
    color: WHITE, margin: 0,
  });

  const lessons = [
    ["측정을 먼저 의심한다", "지표가 상식을 벗어나면 개선보다 검증이 먼저다. 알고리즘을 세 번 바꿔도 수치는 움직이지 않았다."],
    ["기준선 없이는 평가가 없다", "TF-IDF 0.605를 먼저 확보한 뒤에야 0.673이 의미 있는 향상인지 판단할 수 있었다."],
    ["가진 정답을 버리지 않는다", "분야 라벨이 이미 있었는데 \"의미 없을 것\"이라 판단한 것이 어긋난 분기점이었다."],
  ];
  lessons.forEach((l, i) => {
    const y = 1.62 + i * 1.32;
    s.addText(String(i + 1), {
      x: 0.9, y: y, w: 0.5, h: 0.5, fontFace: KR, fontSize: 24, bold: true,
      color: ORANGE, margin: 0,
    });
    s.addText(l[0], {
      x: 1.5, y: y, w: 10.9, h: 0.42, fontFace: KR, fontSize: 20, bold: true,
      color: WHITE, margin: 0,
    });
    s.addText(l[1], {
      x: 1.5, y: y + 0.46, w: 10.9, h: 0.6, fontFace: KR, fontSize: 14,
      color: "AEBACC", margin: 0, valign: "top",
    });
  });

  s.addText("직접 써 보기", {
    x: 0.9, y: 5.72, w: 5.0, h: 0.35, fontFace: KR, fontSize: 14, bold: true,
    color: ORANGE, margin: 0,
  });
  s.addText("nk562220.github.io/re-analysis-of-korscideberta", {
    x: 0.9, y: 6.08, w: 8.0, h: 0.42, fontFace: "Calibri", fontSize: 17, bold: true,
    color: WHITE, margin: 0,
  });
  s.addText("초록을 붙여넣으면 분야를 예측하는 웹 분류기\n(브라우저에서 직접 추론 — 서버 없음)", {
    x: 9.1, y: 5.72, w: 3.6, h: 0.8, fontFace: KR, fontSize: 12,
    color: "AEBACC", margin: 0, align: "right", lineSpacing: 17,
  });
  s.addNotes("모델은 브라우저에서 직접 실행되므로 서버 비용이 없고 입력한 초록이 외부로 나가지 않습니다. 발표 후 직접 시연할 수 있습니다.");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("생성 완료:", f));
