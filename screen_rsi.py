#!/usr/bin/env python3
"""매도 소진 스캐너: RSI 과매도 + 거래량 연속 감소 종목 찾기 → 텔레그램.

조건 (config.json의 "screen" 항목으로 변경 가능)
- 주가 min_price ~ max_price (기본 $5~$40), 미국 정규 거래소 전체
- RSI(rsi_period, 기본 14) <= rsi_max (기본 30)
- 최근 vol_days(기본 5)거래일 거래량이 매일 줄어듦 (5일 전 > 4일 전 > ... > 1일 전)

장 마감 후 하루 1회 실행.
  python screen_rsi.py            # 뉴욕 16:15~18:30 사이에만, 하루 1회
  python screen_rsi.py --force    # 시간·중복 무시
  python screen_rsi.py --dry-run  # 전송·저장 없이 화면 출력
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

DEF = {"min_price": 5, "max_price": 40, "rsi_period": 14, "rsi_max": 30, "vol_days": 5, "max_list": 40}
SC = {**DEF, **al.CONFIG.get("screen", {})}
STATE = al.ROOT / "state_screen.json"
OUT = al.ROOT / "results" / "screen"


def in_window(now: datetime) -> bool:
    return now.weekday() < 5 and dtime(16, 15) <= now.time() <= dtime(18, 30)


def universe() -> dict[str, str]:
    """가격 범위 안의 미국 정규 거래소 종목 전부 (Yahoo 스크리너 250개씩 페이지 조회)."""
    import yfinance as yf
    from yfinance import EquityQuery as Q

    q = Q("and", [Q("eq", ["region", "us"]), Q("is-in", ["exchange", *al.CONFIG["exchanges"]]),
                  Q("gte", ["intradayprice", SC["min_price"]]), Q("lte", ["intradayprice", SC["max_price"]])])
    names, off = {}, 0
    while off < 8000:
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


def scan(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    k = SC["vol_days"]
    r = rsi(close, SC["rsi_period"])
    rows = []
    for s in close.columns:
        c, v = close[s].dropna(), vol[s].reindex(close[s].dropna().index)
        if len(c) < max(SC["rsi_period"] * 3, 61) or v.iloc[-k:].isna().any():
            continue
        px, rv = float(c.iloc[-1]), float(r[s].loc[c.index[-1]])
        if not (SC["min_price"] <= px <= SC["max_price"]) or not (rv <= SC["rsi_max"]):
            continue
        last = v.iloc[-k:].to_numpy(dtype=float)
        if not (np.all(np.diff(last) < 0) and last[-1] > 0):
            continue
        hi60 = float(c.iloc[-60:].max())
        rows.append({"symbol": s, "price": px, "chg": (px / float(c.iloc[-2]) - 1) * 100, "rsi": rv,
                     "from_hi60": (px / hi60 - 1) * 100, "vol_drop": (last[-1] / last[0] - 1) * 100,
                     "vol_vs_avg": (last[-1] / float(v.iloc[-21:-1].mean()) - 1) * 100,
                     "date": c.index[-1].strftime("%Y-%m-%d")})
    return pd.DataFrame(rows).sort_values("rsi") if rows else pd.DataFrame()


def message(df: pd.DataFrame, n_univ: int, day: str, meta: dict) -> str:
    L = [f"🔎 매도 소진 스캐너 {day[5:].replace('-', '/')} (장 마감 기준)",
         f"조건: RSI({SC['rsi_period']}) ≤ {SC['rsi_max']} · 거래량 {SC['vol_days']}일 연속 감소 · "
         f"${SC['min_price']}~${SC['max_price']}",
         f"검색 {n_univ:,}종목 → {len(df)}종목 해당", ""]
    if df.empty:
        L.append("오늘은 조건에 맞는 종목이 없어요.")
    for r in df.head(SC["max_list"]).itertuples():
        L += [f"• {al.label(r.symbol, meta)}",
              f"   ${r.price:.2f} ({r.chg:+.1f}%)  RSI {r.rsi:.0f}  60일 고점 대비 {r.from_hi60:+.0f}%",
              f"   거래량 {SC['vol_days']}일간 {r.vol_drop:+.0f}% (20일 평균 대비 {r.vol_vs_avg:+.0f}%)"]
    if len(df) > SC["max_list"]:
        L.append(f"… 외 {len(df) - SC['max_list']}종목 (RSI 낮은 순 상위 {SC['max_list']}개만 표시)")
    L += ["", "※ 반등을 보장하는 신호가 아니에요. 참고용으로만 보세요."]
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
    print(f"대상 {len(names)}종목")
    close, vol = download(sorted(names))
    day = close.index[-1].strftime("%Y-%m-%d")
    df = scan(close, vol)
    print(f"기준일 {day}, 해당 {len(df)}종목: {', '.join(df['symbol']) if len(df) else '-'}")

    meta = al.load_meta()
    al.fill_meta(meta, list(df["symbol"].head(SC["max_list"])) if len(df) else [], names)
    msg = message(df, len(names), day, meta)
    if args.dry_run:
        print("\n" + msg)
        return 0
    al.send_telegram(msg)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{day}.csv", index=False)
    STATE.write_text(json.dumps({"last_run": now.strftime("%Y-%m-%d"), "last_day": day}), encoding="utf-8")
    al.META_PATH.write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
