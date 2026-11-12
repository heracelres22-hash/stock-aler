#!/usr/bin/env python3
"""소수 종목 포트폴리오 백테스트: 5일선 돌파 매수 → 5일선 이탈 매도, 최대 N종목.

변형 비교
  A: 필터 없음, 우선순위 = 당일 상승률
  B: A + 추세 필터 (종가 > 40주선)
  C: B + 시장 필터 (SPY 종가 > SPY 40주선일 때만 신규 매수)
  D: C, 우선순위 = 6개월(126거래일) 상승률

체결: 신호일 종가. 매도 먼저 처리 후 빈 자리만큼 매수. 각 종목 = 매수 시점 총자산 / N.
비용: 매수·매도 각각 COST (기본 0.1%).
python portfolio.py → results/portfolio.md, results/portfolio_trades_*.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import backtest as bt

MAX_POS = 5
COST = 0.001
VARIANTS = {
    "A": {"trend": False, "market": False, "rank": "day", "desc": "필터 없음 · 당일 상승률 순"},
    "B": {"trend": True, "market": False, "rank": "day", "desc": "40주선 위 종목만 · 당일 상승률 순"},
    "C": {"trend": True, "market": True, "rank": "day", "desc": "B + SPY 40주선 위일 때만 매수"},
    "D": {"trend": True, "market": True, "rank": "mom6", "desc": "C + 6개월 상승률 순"},
    "E": {"trend": True, "market": True, "rank": "mom6", "exit": "ma20", "desc": "D 매수 · 20일선 이탈 매도"},
    "F": {"trend": True, "market": True, "rank": "mom6", "exit": "ma40", "desc": "D 매수 · 40주선 이탈 매도"},
    "G": {"entry": "long", "trend": False, "market": False, "rank": "mom6", "exit": "ma40",
          "desc": "40주선 돌파 매수 · 40주선 이탈 매도 · 6개월 상승률 순"},
}


def run(v, close, eligible, sig, spy_ok, start):
    entry = sig[v.get("entry", "short")][0]
    exit_rule = v.get("exit", "ma5")
    if exit_rule == "ma5":
        below = sig["short"][1]
    elif exit_rule == "ma20":
        below = close < close.rolling(20).mean()
    else:
        below = sig["long"][1]
    cand = entry & eligible
    if v["trend"]:
        cand &= close > sig["ma40"]
    score = close.pct_change(fill_method=None) if v["rank"] == "day" else close / close.shift(126) - 1

    dates, cols = close.index, list(close.columns)
    C, CAND, SC, BEL = close.values, cand.fillna(False).values, score.values, below.fillna(False).values
    mkt = spy_ok.reindex(dates).fillna(False).values
    i0 = dates.searchsorted(pd.Timestamp(start))

    cash, pos = 1.0, {}   # pos[j] = [shares, entry_i, entry_px]
    equity, trades = [], []
    for i in range(i0, len(dates)):
        px = C[i]
        # 매도: 매수 다음 날부터, 종가 < 기준 이평 (데이터 끊김도 매도)
        for j in list(pos):
            sh, ei, ep = pos[j]
            p = px[j] if not np.isnan(px[j]) else C[i - 1, j]
            if i > ei and (BEL[i, j] or np.isnan(px[j])):
                cash += sh * p * (1 - COST)
                trades.append((cols[j], dates[ei], ep, dates[i], p, i - ei, False))
                del pos[j]
        # 매수
        free = MAX_POS - len(pos)
        if free > 0 and (mkt[i] or not v["market"]):
            js = [j for j in np.flatnonzero(CAND[i]) if j not in pos and not np.isnan(SC[i, j])]
            js.sort(key=lambda j: SC[i, j], reverse=True)
            total = cash + sum(sh * (px[j] if not np.isnan(px[j]) else C[i - 1, j]) for j, (sh, _, _) in pos.items())
            for j in js[:free]:
                amt = min(total / MAX_POS, cash)
                if amt <= 0:
                    break
                pos[j] = [amt * (1 - COST) / px[j], i, px[j]]
                cash -= amt
        val = cash + sum(sh * (px[j] if not np.isnan(px[j]) else C[i - 1, j]) for j, (sh, _, _) in pos.items())
        equity.append(val)
    for j, (sh, ei, ep) in pos.items():  # 보유 중
        trades.append((cols[j], dates[ei], ep, dates[-1], C[-1, j], len(dates) - 1 - ei, True))
    eq = pd.Series(equity, index=dates[i0:])
    t = pd.DataFrame(trades, columns=["symbol", "entry_date", "entry", "exit_date", "exit", "days", "open"])
    t["ret"] = (t["exit"] * (1 - COST)) / (t["entry"] / (1 - COST)) - 1
    return eq, t


def stats(eq: pd.Series, t: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    rows = []
    prev = 1.0
    for y in sorted(set(eq.index.year)):
        e = eq[eq.index.year == y]
        full = pd.concat([pd.Series([prev]), e.reset_index(drop=True)])
        mdd = (full / full.cummax() - 1).min()
        s = spy[(spy.index.year == y)].dropna()
        tr = t[t["entry_date"].dt.year == y]
        rows.append({"연도": y, "수익률": (e.iloc[-1] / prev - 1) * 100, "MDD": mdd * 100,
                     "SPY": (s.iloc[-1] / s.iloc[0] - 1) * 100, "거래수": len(tr),
                     "승률": (tr["ret"] > 0).mean() * 100 if len(tr) else np.nan})
        prev = e.iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    s = spy[spy.index >= eq.index[0]].dropna()
    rows.append({"연도": "전체", "수익률": (eq.iloc[-1] - 1) * 100,
                 "MDD": ((eq / eq.cummax()) - 1).min() * 100,
                 "SPY": (s.iloc[-1] / s.iloc[0] - 1) * 100, "거래수": len(t),
                 "승률": (t["ret"] > 0).mean() * 100,
                 "CAGR": (eq.iloc[-1] ** (1 / years) - 1) * 100,
                 "평균보유일": t["days"].mean()})
    return pd.DataFrame(rows)


def main():
    bt.OUT.mkdir(exist_ok=True)
    pool = bt.candidate_pool()
    close, vol = bt.load_prices(pool)
    spy = close.pop("SPY")
    close = close[vol.columns]
    rank = vol.where((close >= bt.CFG["min_price"]) & close.notna()).rank(axis=1, ascending=False, method="first")
    eligible = rank <= bt.CFG["top_n"]
    sig = bt.signals(close)
    spy_ma40 = bt.signals(spy.to_frame())["ma40"]["SPY"]
    spy_ok = spy > spy_ma40

    s_spy = spy[spy.index >= bt.START].dropna()
    spy_cagr = ((s_spy.iloc[-1] / s_spy.iloc[0]) ** (365.25 / (s_spy.index[-1] - s_spy.index[0]).days) - 1) * 100
    spy_mdd = ((s_spy / s_spy.cummax()) - 1).min() * 100

    md = [f"# 소수 종목 백테스트 (최대 {MAX_POS}종목, {bt.START[:4]} ~ {close.index[-1].date()})", "",
          "- 매수: 이평 상향 돌파 (그날 거래량 상위 100, 주가 $20 이상) · 매도: 종가 < 기준 이평 (방식별)",
          f"- 각 종목 = 총자산의 1/{MAX_POS}, 신호일 종가 체결, 비용 매수·매도 각 {COST*100:.1f}%",
          f"- 후보군 = 현재 거래량 상위 {len(pool)}종목 (생존 편향 있음)", "", "## 요약", "",
          "| 방식 | 설명 | 누적수익 | 연평균(CAGR) | 최대낙폭 | 거래수 | 승률 | 평균보유일 |",
          "|---|---|---|---|---|---|---|---|"]
    detail, years = [], {}
    for k, v in VARIANTS.items():
        eq, t = run(v, close, eligible, sig, spy_ok, bt.START)
        t.to_csv(bt.OUT / f"portfolio_trades_{k}.csv", index=False)
        st = stats(eq, t, spy)
        tot = st.iloc[-1]
        md.append(f"| {k} | {v['desc']} | {tot['수익률']:+.0f}% | {tot['CAGR']:+.1f}% | {tot['MDD']:.1f}% | "
                  f"{tot['거래수']:.0f} | {tot['승률']:.1f}% | {tot['평균보유일']:.1f} |")
        years[k] = st.set_index("연도")["수익률"]
        print(k, v["desc"]); print(st.to_string())
    md.append(f"| SPY | 보유만 | {(s_spy.iloc[-1]/s_spy.iloc[0]-1)*100:+.0f}% | {spy_cagr:+.1f}% | {spy_mdd:.1f}% | - | - | - |")
    yt = pd.DataFrame(years)
    yt["SPY"] = stats(eq, t, spy).set_index("연도")["SPY"]
    md += ["", "## 연도별 수익률", "", "| 연도 | " + " | ".join(yt.columns) + " |", "|" + "---|" * (len(yt.columns) + 1)]
    for y, r in yt.iterrows():
        md.append(f"| {y} | " + " | ".join(f"{x:+.1f}%" for x in r) + " |")
    (bt.OUT / "portfolio.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
