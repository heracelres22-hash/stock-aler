#!/usr/bin/env python3
"""횡보 후 5일선 돌파 스캐너 → 텔레그램 (장 마감 후 하루 1회).

1) 조건 (config.json "screen"): 주가 $100~$200, 직전 range_days(40)거래일
   고가-저가 범위가 range_pct(10)% 이내로 횡보 + 당일 5일선 상향 돌파 → 🟢 알림
2) 추적 종목이 하락 기준선(20일선 -5%) 아래로 마감 → 🔴 알림 (따로 전송)
   돌파 때 이미 기준선 아래였던 종목은, 기준선 위로 올라오기 전에는 돌파 가격 아래로 내려갈 때 알림
  python screen_rsi.py [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, time as dtime

import pandas as pd

import alert as al

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
                names[x["symbol"]] = x.get("longName") or x.get("shortName") or ""
            if len(quotes) < 250:
                break
            off += 250
    return names


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
    """날짜×종목 매수 신호: 가격대 + 직전 range_days거래일 횡보(고저 범위 ≤range_pct%) + 당일 5일선 상향 돌파."""
    rd = SC["range_days"]
    hi = close.rolling(rd).max().shift(1)
    lo = close.rolling(rd).min().shift(1)
    rng_ok = ((hi - lo) / hi * 100 <= SC["range_pct"]) & hi.notna()
    band = (close >= SC["min_price"]) & (close <= SC["max_price"])
    ma5 = close.rolling(5).mean()
    cross = (close.shift(1) <= ma5.shift(1)) & (close > ma5)
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
    L = [f"🟢 횡보 후 5일선 돌파 {fmt(day)} (장 마감 기준)",
         f"섹터: S&P 500 시가총액 상위 2개 (Technology · Financial Services)",
         f"조건: ${SC['min_price']}~${SC['max_price']} · 직전 {SC['range_days']}거래일 고저 범위 ≤{SC['range_pct']:g}% 횡보 → 5일선 상향 돌파",
         f"가치 필터: PER≤{SC['max_per']:g} · 부채비율≤{SC['max_debt_equity']:g}% · ROE≥{SC['min_roe']:g}%", ""]
    for b in buys:
        L += [f"• {al.label(b['symbol'], meta)}",
              f"   ${b['price']:.2f} ({b['chg']:+.1f}%)  20일선 ${b['ma20']:.2f}  "
              f"60일 고점 대비 {b['from_hi60']:+.0f}%"]
    L += ["", "※ 반등을 보장하는 신호가 아니에요. 참고용으로만 보세요."]
    return "\n".join(L)


def exit_message(day, exits, expired, meta) -> str:
    L = [f"🔴 하락 알림 {fmt(day)} (5일선 돌파 알림 종목 · 기준: 20일선 -{SC['exit_pct']:g}%)", ""]
    for e in exits:
        L += [f"• {al.label(e['symbol'], meta)}",
              f"   ${e['price']:.2f}  기준선 ${e['line']:.2f} (20일선 ${e['ma20']:.2f})  · {e['why']}",
              f"   돌파 {fmt(e['e_since'])} ${e['e_price']:.2f} → {(e['price']/e['e_price']-1)*100:+.1f}%"]
    if expired:
        L += ["", f"⏹ 추적 종료 ({SC['track_days']}거래일 동안 20일선 위로 못 올라옴)"]
        L += [f"• {s}  ${p['price']:.2f} → ${px:.2f} ({(px/p['price']-1)*100:+.1f}%)" for s, p, px in expired]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
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
