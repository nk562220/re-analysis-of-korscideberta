# -*- coding: utf-8 -*-
"""KorSciDeBERTa 커스텀 토크나이저를 위한 Mecab-ko 준비.

tokenization_korscideberta_v2.py 는 첫 줄에서
    from konlpy.tag import Mecab
을 하고, 내부에서 Mecab().pos(line) 만 사용한다.

원래 설치 경로(konlpy + mecab-ko 소스 빌드)는 Colab에서 자주 깨진다.
그래서 pip 만으로 설치되는 python-mecab-ko(휠에 mecab-ko-dic 포함)를
konlpy.tag.Mecab 자리에 끼워 넣는 shim 을 제공한다.
형태소 분석기 본체와 사전은 동일한 mecab-ko-dic 이므로 분석 결과가 같다.

우선순위:
  1) 정상 설치된 konlpy.tag.Mecab 이 있으면 그것을 쓴다
  2) python-mecab-ko 로 shim 을 만든다   (pip install python-mecab-ko)
  3) 둘 다 실패하면 None 을 반환 -> finetune.py 가 표준 토크나이저로 강등
"""
import sys
import types

_PROBE = "형태소 분석 테스트"


def _works(pos_fn):
    try:
        out = pos_fn(_PROBE)
        return bool(out) and isinstance(out[0], (tuple, list)) and len(out[0]) == 2
    except Exception:
        return False


def ensure_konlpy_mecab(verbose=True):
    """konlpy.tag.Mecab 을 임포트 가능한 상태로 만들고 사용된 백엔드 이름을 반환."""
    # 1) 원래 경로
    try:
        from konlpy.tag import Mecab as _M
        if _works(_M().pos):
            if verbose:
                print("[mecab] konlpy.tag.Mecab (정식 설치) 사용")
            return "konlpy"
    except Exception:
        pass

    # 2) python-mecab-ko 기반 shim
    try:
        import mecab as _pm
        tagger = _pm.MeCab()
        if not _works(tagger.pos):
            raise RuntimeError("python-mecab-ko pos() 동작 실패")
    except Exception as e:
        if verbose:
            print("[mecab] 사용 가능한 Mecab-ko 백엔드 없음 (%s: %s)" % (type(e).__name__, e))
            print("[mecab] 설치: pip install python-mecab-ko   (리눅스/Colab/mac 휠 제공)")
        return None

    class Mecab(object):
        """konlpy.tag.Mecab 의 최소 호환 구현."""

        def __init__(self, *args, **kwargs):
            self._t = _pm.MeCab()

        def pos(self, phrase, flatten=True, join=False):
            pairs = self._t.pos(phrase)
            if join:
                return ["%s/%s" % (s, t) for s, t in pairs]
            return pairs

        def morphs(self, phrase):
            return [s for s, _ in self._t.pos(phrase)]

        def nouns(self, phrase):
            return [s for s, t in self._t.pos(phrase) if t.startswith("N")]

    konlpy = sys.modules.get("konlpy") or types.ModuleType("konlpy")
    tag_mod = types.ModuleType("konlpy.tag")
    tag_mod.Mecab = Mecab
    konlpy.tag = tag_mod
    sys.modules["konlpy"] = konlpy
    sys.modules["konlpy.tag"] = tag_mod

    if verbose:
        print("[mecab] python-mecab-ko 기반 shim 을 konlpy.tag.Mecab 으로 등록")
    return "python-mecab-ko(shim)"


if __name__ == "__main__":
    backend = ensure_konlpy_mecab()
    if backend:
        from konlpy.tag import Mecab
        print("백엔드:", backend)
        print(Mecab().pos("이산화티타늄 광촉매를 이용한 미세플라스틱 제거 연구"))
    else:
        sys.exit(1)
