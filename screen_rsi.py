#!/usr/bin/env python3
"""횡보 후 20일선 돌파 스캐너 → 텔레그램 (장 마감 후 하루 1회).

1) 조건 (config.json "screen"): 주가 $100~$200, 직전 range_days(40)거래일
   고가-저가 범위가 range_pct(10)% 이내로 횡보 + 당일 20일선 상향 돌파 → 🟢 알림
2) 추적 종목이 하락 기준선(20일선 -5%) 아래로 마감 → 🔴 알림 (따로 전송)
   돌파 때 이미 기준선 아래였던 종목은, 기준선 위로 올라오기 전에는 돌파 가격 아래로 내려갈 때 알림
  python screen_rsi.py [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, time as dtime

import pandas as pd

import alert as al

# 펀드·신탁·SPAC 등 사업회사가 아닌 종목을 이름/사업설명으로 식별
# ("Northern Trust" 같은 정상 기업이 걸리지 않도록 trust는 수식어와 함께만 매칭)
FUND_RE = re.compile(
    r"\b(fund|funds|etf|etn|closed[-\s]?end|spac|acquisition corp|blank check"
    r"|(income|credit|royalty|unit|investment|mortgage|realty|properties)\s+trust)\b", re.I)

DEF = {"min_price": 100, "max_price": 200, "range_days": 40, "range_pct": 10, "track_days": 60,
       "exit_pct": 5, "min_mktcap": 1_000_000_000,
       "max_per": 25, "max_debt_equity": 150, "min_roe": 10}
SC = {**DEF, **al.CONFIG.get("screen", {})}
STATE = al.ROOT / "state_screen.json"
OUT = al.ROOT / "results" / "screen"


def in_window(now: datetime) -> bool:
    return now.weekday() < 5 and dtime(16, 15) <= now.time() <= dtime(18, 30)


def universe() -> dict[str, str]:
    """S&P 500 시가총액 상위 2개 섹터(Technology, Financial Services)에서
    가격 범위 안의 미국 정규 거래소 종목만 조회 (Yahoo 스크리너 250개씩 페이지 조회)."""
    import yfinance as yf
    from yfinance import EquityQuery as Q

    # S&P 500 시가총액 상위 2개 섹터 (yfinance 스크리너 섹터 값 기준)
    TOP_SECTORS = ["Technology", "Financial Services"]

    names: dict[str, str] = {}
    for sector in TOP_SECTORS:
        off = 0
        while off < 8000:
            q = Q("and", [
                Q("eq", ["region", "us"]),
                Q("is-in", ["exchange", *al.CONFIG["exchanges"]]),
                Q("gte", ["intradayprice", SC["min_price"]]),
                Q("lte", ["intradayprice", SC["max_price"]]),
                Q("eq", ["sector", sector]),
                Q("gte", ["intradaymarketcap", SC["min_mktcap"]]),
                # 그레이엄/버핏식 퀄리티-가치 필터: 저평가 + 낮은 부채 + 높은 ROE
                Q("lte", ["peratio.lasttwelvemonths", SC["max_per"]]),
                Q("lte", ["totaldebtequity.lasttwelvemonths", SC["max_debt_equity"]]),
                Q("gte", ["returnonequity.lasttwelvemonths", SC["min_roe"]]),
            ])
            res = yf.screen(q, sortField="dayvolume", sortAsc=False, size=250, offset=off)
            # SPAC 유닛/워런트/우선주/신탁 티커(대시 포함) 제외 — 정상 보통주만
            quotes = [x for x in res.get("quotes", []) if x.get("symbol") and "-" not in x["symbol"]]
            for x in quotes:
                name = x.get("longName") or x.get("shortName") or ""
                qt = (x.get("quoteType") or "").upper()
                if qt and qt != "EQUITY":            # ETF·펀드 등 주식이 아닌 종목
                    continue
                if FUND_RE.search(name):             # 폐쇄형 펀드·신탁·SPAC
                    continue
                names[x["symbol"]] = name
            if len(quotes) < 250:
                break
            off += 250
    return names


def _tr_deepl(text: str) -> str:
    """DeepL (DEEPL_API_KEY 시크릿이 있을 때만). 품질이 가장 좋음."""
    import requests

    key = os.environ.get("DEEPL_API_KEY")
    if not key:
        raise RuntimeError("키 없음")
    host = "api-free.deepl.com" if key.endswith(":fx") else "api.deepl.com"
    r = requests.post(f"https://{host}/v2/translate",
                      data={"text": text, "source_lang": "EN", "target_lang": "KO"},
                      headers={"Authorization": f"DeepL-Auth-Key {key}"}, timeout=12)
    r.raise_for_status()
    return (r.json()["translations"][0]["text"] or "").strip()


def _tr_mymemory(text: str) -> str:
    """MyMemory — 키 없이 사용 가능 (익명 한도 하루 약 5천 자)."""
    import requests

    r = requests.get("https://api.mymemory.translated.net/get",
                     params={"q": text, "langpair": "en|ko"}, timeout=12)
    r.raise_for_status()
    j = r.json()
    out = ((j.get("responseData") or {}).get("translatedText") or "").strip()
    if not out or "MYMEMORY WARNING" in out.upper() or "QUERY LENGTH LIMIT" in out.upper():
        raise RuntimeError(f"응답 이상({j.get('responseStatus')})")
    return out


def _tr_google(text: str) -> str:
    """구글 비공식 엔드포인트 — 데이터센터 IP는 차단될 수 있음."""
    import requests

    r = requests.get("https://translate.googleapis.com/translate_a/single",
                     params={"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "q": text},
                     timeout=10)
    r.raise_for_status()
    return "".join(p[0] for p in r.json()[0] if p and p[0]).strip()


TRANSLATORS = [("DeepL", _tr_deepl), ("MyMemory", _tr_mymemory), ("Google", _tr_google)]
_TR_DEAD: set[str] = set()      # 이번 실행에서 이미 실패한 제공처는 다시 시도하지 않음


def translate_ko(text: str) -> str:
    """영문 헤드라인을 한국어로. 모두 실패하면 원문 그대로 (알림이 끊기지 않도록)."""
    if not text:
        return text
    for name, fn in TRANSLATORS:
        if name in _TR_DEAD:
            continue
        try:
            out = fn(text)
            if out:
                return out
            raise RuntimeError("빈 응답")
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            print(f"[번역 실패:{name}] {code or type(e).__name__} {e}"[:120])
            _TR_DEAD.add(name)
    return text


def recent_news(sym: str, count: int = 3, days: int = 14) -> list[str]:
    """최근 뉴스 헤드라인 (한글 제목 · 출처 · 날짜 · 링크). 실패하면 빈 목록."""
    import yfinance as yf

    try:
        items = yf.Ticker(sym).get_news(count=10, tab="news") or []
    except Exception as e:
        print(f"[뉴스 실패] {sym}: {type(e).__name__}")
        return []
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    out = []
    for it in items:
        c = it.get("content") or {}
        title = (c.get("title") or "").strip()
        if not title:
            continue
        when = pd.to_datetime(c.get("pubDate") or c.get("displayTime"), errors="coerce", utc=True)
        if pd.notna(when) and when < cutoff:
            continue
        src = ((c.get("provider") or {}).get("displayName") or "").strip()
        url = ((c.get("canonicalUrl") or c.get("clickThroughUrl") or {}) or {}).get("url") or ""
        title = translate_ko(title)
        if len(title) > 90:
            title = title[:89].rstrip() + "…"
        head = f"   - {title}"
        tail = " · ".join(x for x in (src, when.strftime("%m/%d") if pd.notna(when) else "") if x)
        out.append(f"{head}\n     {tail}\n     {url}".rstrip())
        if len(out) >= count:
            break
    return out


def is_operating_company(sym: str) -> tuple[bool, str]:
    """알림 직전 최종 확인: 펀드·신탁처럼 사업회사가 아닌 종목이면 (False, 이유).
    조회가 실패하면 통과시킨다 (일시적 오류로 진짜 신호를 놓치지 않도록)."""
    import yfinance as yf

    try:
        info = yf.Ticker(sym).get_info() or {}
    except Exception as e:
        return True, f"확인 실패({type(e).__name__})"
    qt = (info.get("quoteType") or "").upper()
    if qt and qt != "EQUITY":
        return False, f"quoteType={qt}"
    name = f"{info.get('longName') or ''} {info.get('shortName') or ''}"
    if FUND_RE.search(name):
        return False, "이름이 펀드·신탁"
    summary = info.get("longBusinessSummary") or ""
    if FUND_RE.search(summary) and not info.get("fullTimeEmployees"):
        return False, "사업설명이 펀드 + 직원수 없음"
    return True, ""


def download(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf

    closes = []
    for i in range(0, len(symbols), 400):
        chunk = symbols[i:i + 400]
        raw = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                          auto_adjust=True, threads=True, progress=False)
        lv = raw.columns.levels[0] if isinstance(raw.columns, pd.MultiIndex) else []
        closes.append(pd.DataFrame({s: raw[s]["Close"] for s in chunk if s in lv}))
    close = pd.concat(closes, axis=1)
    idx = pd.DatetimeIndex(close.index)
    close.index = idx.tz_localize(None) if idx.tz is not None else idx
    return close.sort_index()


def setups(close: pd.DataFrame) -> pd.DataFrame:
    """날짜×종목 매수 신호: 가격대 + 직전 range_days거래일 횡보(고저 범위 ≤range_pct%) + 당일 20일선 상향 돌파."""
    rd = SC["range_days"]
    hi = close.rolling(rd).max().shift(1)
    lo = close.rolling(rd).min().shift(1)
    rng_ok = ((hi - lo) / hi * 100 <= SC["range_pct"]) & hi.notna()
    band = (close >= SC["min_price"]) & (close <= SC["max_price"])
    ma20 = close.rolling(20).mean()
    cross = (close.shift(1) <= ma20.shift(1)) & (close > ma20)
    return band & rng_ok & cross


def fmt(d: str) -> str:
    return d[5:].replace("-", "/")


def run_day(close: pd.DataFrame, st: dict):
    """오늘 기준 돌파·이탈 판정. st(active, alerted)를 갱신하고 (day, buys, exits, expired) 반환."""
    day = close.index[-1].strftime("%Y-%m-%d")
    sig = setups(close)
    ma20 = close.rolling(20).mean()
    active, alerted = st.setdefault("active", {}), st.setdefault("alerted", {})

    exits, expired = [], []
    for s, p in list(active.items()):
        if s not in close.columns or pd.isna(close[s].iloc[-1]) or day <= p["since"]:
            continue
        px, m20 = float(close[s].iloc[-1]), float(ma20[s].iloc[-1])
        line = m20 * (1 - SC["exit_pct"] / 100)      # 하락 기준선: 20일선 -5%
        if not p.get("armed") and px >= line:
            p["armed"] = True                         # 기준선 위로 올라옴 → 이후 기준선 이탈 시 알림
            continue
        # 기준선 위로 올라온 적 있으면: 기준선 아래로 내려갈 때 / 없으면: 돌파 가격 아래로 내려갈 때
        if (p.get("armed") and px < line) or (not p.get("armed") and px < p["price"]):
            exits.append({"symbol": s, "price": px, "ma20": m20, "line": line,
                          "why": "기준선 이탈" if p.get("armed") else "돌파가 아래로 하락",
                          **{f"e_{k}": v for k, v in p.items()}})
            del active[s]

    buys = []
    today_sig = sig.columns[sig.iloc[-1].fillna(False)] if len(sig) else []
    for s in today_sig:
        if s in active or alerted.get(s) == day:
            continue
        c = close[s]
        if pd.isna(c.iloc[-1]) or pd.isna(c.iloc[-2]):
            continue
        px, m20 = float(c.iloc[-1]), float(ma20[s].iloc[-1])
        buys.append({"symbol": s, "price": px, "chg": (px / float(c.iloc[-2]) - 1) * 100,
                     "ma20": m20, "from_hi60": (px / float(c.iloc[-60:].max()) - 1) * 100})
        active[s] = {"since": day, "price": round(px, 4), "armed": bool(px >= m20 * (1 - SC["exit_pct"] / 100))}
        alerted[s] = day
    buys.sort(key=lambda b: b["chg"], reverse=True)
    return day, buys, exits, expired


def buy_message(day, buys, meta) -> str:
    L = [f"🟢 횡보 후 20일선 돌파 {fmt(day)} (장 마감 기준)",
         f"섹터: S&P 500 시가총액 상위 2개 (Technology · Financial Services)",
         f"조건: ${SC['min_price']}~${SC['max_price']} · 직전 {SC['range_days']}거래일 고저 범위 ≤{SC['range_pct']:g}% 횡보 → 20일선 상향 돌파",
         f"가치 필터: PER≤{SC['max_per']:g} · 부채비율≤{SC['max_debt_equity']:g}% · ROE≥{SC['min_roe']:g}%", ""]
    for b in buys:
        L += [f"• {al.label(b['symbol'], meta)}",
              f"   ${b['price']:.2f} ({b['chg']:+.1f}%)  20일선 ${b['ma20']:.2f}  "
              f"60일 고점 대비 {b['from_hi60']:+.0f}%"]
        news = b.get("news") or []
        L += ["   📰 최근 뉴스", *news] if news else ["   📰 최근 2주 뉴스 없음"]
        L.append("")
    L += ["※ 반등을 보장하는 신호가 아니에요. 참고용으로만 보세요."]
    return "\n".join(L)


def exit_message(day, exits, expired, meta) -> str:
    L = [f"🔴 하락 알림 {fmt(day)} (20일선 돌파 알림 종목 · 기준: 20일선 -{SC['exit_pct']:g}%)", ""]
    for e in exits:
        L += [f"• {al.label(e['symbol'], meta)}",
              f"   ${e['price']:.2f}  기준선 ${e['line']:.2f} (20일선 ${e['ma20']:.2f})  · {e['why']}",
              f"   돌파 {fmt(e['e_since'])} ${e['e_price']:.2f} → {(e['price']/e['e_price']-1)*100:+.1f}%"]
    if expired:
        L += ["", f"⏹ 추적 종료 ({SC['track_days']}거래일 동안 20일선 위로 못 올라옴)"]
        L += [f"• {s}  ${p['price']:.2f} → ${px:.2f} ({(px/p['price']-1)*100:+.1f}%)" for s, p, px in expired]
    return "\n".join(L)


def replay(n: int) -> int:
    """최근 n거래일을 현재 규칙으로 되돌려 매수 신호를 나열 (전송·상태 저장 없음).
    setups()는 과거 데이터만 쓰므로 그날 알림이 봤을 것과 같은 결과."""
    names = universe()
    syms = sorted(names)
    print(f"대상 {len(syms)}종목 · 최근 {n}거래일 리플레이")
    close = download(syms).dropna(how="all")
    sig = setups(close).iloc[-n:]
    checked: dict[str, tuple[bool, str]] = {}
    total = 0
    for d, row in sig.iterrows():
        hits = [s for s in sig.columns[row.fillna(False)]]
        shown = []
        for s in hits:
            if s not in checked:
                checked[s] = is_operating_company(s)
            ok, why = checked[s]
            px = close[s].loc[d]
            shown.append(f"{s} ${px:.2f}" if ok else f"{s}(제외:{why})")
        ok_cnt = sum(1 for s in hits if checked[s][0])
        total += ok_cnt
        print(f"  {d.date()} · {ok_cnt}건  {', '.join(shown) or '-'}")
    uniq = sorted({s for s in checked if checked[s][0]})
    print(f"\n합계 {total}건 · 중복 제외 {len(uniq)}종목: {', '.join(uniq) or '-'}")
    if not uniq:
        return 0

    # 실제 전송될 메시지 미리보기 (신호가 있던 마지막 날 기준)
    last = max(d for d, row in sig.iterrows() if any(checked.get(s, (False,))[0]
                                                     for s in sig.columns[row.fillna(False)]))
    i = close.index.get_loc(last)
    ma20 = close.rolling(20).mean()
    buys = []
    for s in sig.columns[sig.loc[last].fillna(False)]:
        if not checked.get(s, (False,))[0]:
            continue
        px = float(close[s].iloc[i])
        buys.append({"symbol": s, "price": px, "chg": (px / float(close[s].iloc[i - 1]) - 1) * 100,
                     "ma20": float(ma20[s].iloc[i]),
                     "from_hi60": (px / float(close[s].iloc[max(0, i - 59):i + 1].max()) - 1) * 100,
                     "news": recent_news(s)})
    meta = al.load_meta()
    al.fill_meta(meta, [b["symbol"] for b in buys], names)
    print("\n--- 전송 메시지 미리보기 ---")
    print(buy_message(last.strftime("%Y-%m-%d"), buys, meta))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", type=int, default=0, help="최근 N거래일 신호만 조회 (전송 없음)")
    args = ap.parse_args()
    probe = os.environ.get("TRANSLATE_TEST")
    if probe:                                        # 번역 제공처 점검 (데이터 조회 없이 즉시)
        for name, fn in TRANSLATORS:
            try:
                print(f"  {name}: {fn(probe)}")
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                print(f"  {name}: 실패 — {code or type(e).__name__} {e}"[:160])
        return 0
    n = args.replay or int(os.environ.get("REPLAY") or 0)
    if n:
        return replay(n)
    force = args.force or os.environ.get("FORCE") == "1"
    now = datetime.now(al.NY)
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if not force and (not in_window(now) or st.get("last_run") == now.strftime("%Y-%m-%d")):
        print(f"실행 시간이 아니거나 오늘 이미 실행함 ({now:%a %H:%M} ET) → 건너뜀")
        return 0

    names = universe()
    syms = sorted(set(names) | set(st.get("active", {})))
    print(f"대상 {len(syms)}종목")
    close = download(syms)
    close = close.dropna(how="all")
    day, buys, exits, expired = run_day(close, st)
    kept = []                                        # 알림 직전 최종 확인: 펀드·신탁 제외
    for b in buys:
        ok, why = is_operating_company(b["symbol"])
        if ok:
            b["news"] = recent_news(b["symbol"])     # 알림에 붙일 최근 뉴스
            kept.append(b)
        else:
            print(f"[제외] {b['symbol']}: {why}")
            st.get("active", {}).pop(b["symbol"], None)
            st.get("alerted", {}).pop(b["symbol"], None)
    buys = kept
    print(f"기준일 {day}")
    print(f"돌파 {[b['symbol'] for b in buys]} · 이탈 {[e['symbol'] for e in exits]} · "
          f"종료 {[x[0] for x in expired]} · 추적 중 {sorted(st['active'])}")

    meta = al.load_meta()
    al.fill_meta(meta, [b["symbol"] for b in buys] + [e["symbol"] for e in exits], names)
    msgs = ([buy_message(day, buys, meta)] if buys else []) + \
           ([exit_message(day, exits, expired, meta)] if exits or expired else [])
    if args.dry_run:
        print("\n" + "\n\n".join(msgs))
        return 0
    for m in msgs:                                   # 돌파와 이탈은 따로 전송
        al.send_telegram(m)
    OUT.mkdir(parents=True, exist_ok=True)
    st.update(last_run=now.strftime("%Y-%m-%d"), last_day=day)
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
    al.META_PATH.write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
