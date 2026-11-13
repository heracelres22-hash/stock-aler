#!/usr/bin/env python3
"""매도 소진 후 반등 스캐너 → 텔레그램 (장 마감 후 하루 1회).

1) 매도 소진 조건 (config.json "screen"): 주가 $5~$40, RSI(14) <= 30,
   거래량 5거래일 연속 감소, 당일 거래량 < 직전 20일 평균
2) 조건 충족 후 watch_days(10)거래일 안에 5일선 상향 돌파 → 🟢 알림, 추적 시작
3) 추적 종목이 하락 기준선(20일선 -5%) 아래로 마감 → 🔴 알림 (따로 전송)
   돌파 때 이미 기준선 아래였던 종목은, 기준선 위로 올라오기 전에는 돌파 가격 아래로 내려갈 때 알림
  python screen_rsi.py [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

import alert as al

DEF = {"min_price": 5, "max_price": 40, "rsi_period": 14, "rsi_max": 30, "vol_days": 5, "watch_days": 10, "track_days": 60, "exit_pct": 5}
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
            ])
            res = yf.screen(q, sortField="dayvolume", sortAsc=False, size=250, offset=off)
            quotes = [x for x in res.get("quotes", []) if x.get("symbol")]
            for x in quotes:
                names[x["symbol"]] = x.get("longName") or x.get("shortName") or ""
            if len(quotes) < 250:
                break
            off += 250
    return names


def download(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    closes, vols = [], []
    for i in range(0, len(symbols), 400):
        chunk = symbols[i:i + 400]
        raw = yf.download(chunk, period="6mo", interval="1d", group_by="ticker",
                          auto_adjust=True, threads=True, progress=False)
        lv = raw.columns.levels[0] if isinstance(raw.columns, pd.MultiIndex) else []
        closes.append(pd.DataFrame({s: raw[s]["Close"] for s in chunk if s in lv}))
        vols.append(pd.DataFrame({s: raw[s]["Volume"] for s in chunk if s in lv}))
    close, vol = pd.concat(closes, axis=1), pd.concat(vols, axis=1)
    idx = pd.DatetimeIndex(close.index)
    close.index = vol.index = idx.tz_localize(None) if idx.tz is not None else idx
    return close.sort_index(), vol.sort_index()


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def setups(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """날짜×종목 매도 소진 여부: 가격대 + RSI 과매도 + 거래량 k일 연속 감소 + 거래량 < 20일 평균."""
    k = SC["vol_days"]
    r = rsi(close, SC["rsi_period"])
    dec = (vol.diff() < 0).rolling(k - 1).sum() == k - 1
    quiet = vol < vol.rolling(20).mean().shift(1)
    band = (close >= SC["min_price"]) & (close <= SC["max_price"])
    return band & (r <= SC["rsi_max"]) & dec & quiet & (vol > 0)


def fmt(d: str) -> str:
    return d[5:].replace("-", "/")


def run_day(close, vol, st):
    """오늘 기준 돌파·이탈 판정. st(active, alerted)를 갱신하고 (buys, exits, expired, watch) 반환."""
    W = SC["watch_days"]
    day = close.index[-1].strftime("%Y-%m-%d")
    su = setups(close, vol).iloc[-W - 1:]            # 오늘 포함 최근 W+1거래일
    r = rsi(close, SC["rsi_period"])
    ma5, ma20 = close.rolling(5).mean(), close.rolling(20).mean()
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

    watch, buys = {}, []
    for s in su.columns[su.any()]:
        sd = su.index[su[s]][-1]                      # 가장 최근 조건 충족일
        sds = sd.strftime("%Y-%m-%d")
        watch[s] = sds
        c = close[s]
        if s in active or alerted.get(s) == sds or pd.isna(c.iloc[-1]) or pd.isna(c.iloc[-2]):
            continue
        if c.iloc[-2] <= ma5[s].iloc[-2] and c.iloc[-1] > ma5[s].iloc[-1]:
            px = float(c.iloc[-1])
            buys.append({"symbol": s, "price": px, "chg": (px / float(c.iloc[-2]) - 1) * 100,
                         "ma5": float(ma5[s].iloc[-1]), "ma20": float(ma20[s].iloc[-1]),
                         "rsi": float(r[s].iloc[-1]), "setup": sds, "setup_rsi": float(r[s].loc[sd]),
                         "from_hi60": (px / float(c.iloc[-60:].max()) - 1) * 100})
            active[s] = {"since": day, "price": round(px, 4), "setup": sds, "armed": bool(px >= ma20[s].iloc[-1] * (1 - SC["exit_pct"] / 100))}
            alerted[s] = sds
    buys.sort(key=lambda b: b["chg"], reverse=True)
    return day, buys, exits, expired, watch


def buy_message(day, buys, meta) -> str:
    L = [f"🟢 매도 소진 후 5일선 돌파 {fmt(day)} (장 마감 기준)",
         f"섹터: S&P 500 시가총액 상위 2개 (Technology · Financial Services)",
         f"조건: RSI({SC['rsi_period']}) ≤ {SC['rsi_max']} · 거래량 {SC['vol_days']}일 연속 감소 · 20일 평균 미만 · "
         f"${SC['min_price']}~${SC['max_price']} → {SC['watch_days']}거래일 안에 5일선 돌파", ""]
    for b in buys:
        L += [f"• {al.label(b['symbol'], meta)}",
              f"   ${b['price']:.2f} ({b['chg']:+.1f}%)  5일선 ${b['ma5']:.2f}  20일선 ${b['ma20']:.2f}",
              f"   조건 충족 {fmt(b['setup'])} (RSI {b['setup_rsi']:.0f}) → 현재 RSI {b['rsi']:.0f}  "
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
    close, vol = download(syms)
    close = close.dropna(how="all")
    vol = vol.reindex(close.index)
    day, buys, exits, expired, watch = run_day(close, vol, st)
    print(f"기준일 {day} · 관찰 {len(watch)}종목: {', '.join(sorted(watch)) or '-'}")
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
    pd.DataFrame({"symbol": list(watch), "setup": list(watch.values())}).to_csv(OUT / f"{day}_watch.csv", index=False)
    st.update(last_run=now.strftime("%Y-%m-%d"), last_day=day)
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
    al.META_PATH.write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
