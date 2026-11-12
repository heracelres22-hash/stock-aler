#!/usr/bin/env python3
"""현재 알림 규칙 백테스트: 5일선 돌파 매수 → 20일선 이탈 매도.

- 대상: 매일 주가 $5 이상 $40 이하 종목 중 그날 거래량 상위 N (config.json top_n)
- 후보군: 현재 Yahoo 거래량 상위 250 ($5~40) + 2014년 이후 S&P 500에 한 번이라도 들어간 종목
  (과거 시점의 전체 미국 주식 목록은 무료로 구할 수 없어 근사. 생존 편향 있음)
- 매수: 5일선 상향 돌파일 종가 (그날 대상에 들어 있을 때) · 종목 수 제한 없음
- 매도: 매수 다음 날부터 종가 < 20일선이 된 날 종가
- 포트폴리오: 계좌를 K칸으로 나눠 신호마다 한 칸(총자산/K)씩 매수, 리밸런싱 없음,
  칸이 다 차면 그날 상승률 높은 순으로만 매수. 비용 매수·매도 각 0.1%
python backtest_ma5_20.py → results/ma5_20.md, results/ma5_20_trades.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import alert as al
import backtest as bt
import portfolio as pf

CFG = bt.CFG
COST = 0.001
LO, HI = CFG["min_price"], CFG["max_price"]
N, ENTRY, EXIT = CFG["top_n"], CFG["short_ma_days"], CFG["exit_ma_days"]


def pool() -> list[str]:
    syms, _ = al.top_by_volume(250, CFG["exchanges"], LO, HI)
    h = pd.read_csv(bt.ROOT / "data" / "sp500_history.csv")
    sp = {t.replace(".", "-") for row in h["tickers"] for t in row.split(",")}
    return sorted(set(syms) | sp)


def trade_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, tr in list(t.groupby(t["entry_date"].dt.year)) + [("전체", t)]:
        rows.append({"연도": y, "거래수": len(tr), "승률": (tr.ret > 0).mean() * 100,
                     "평균수익": tr.ret.mean() * 100, "중간값": tr.ret.median() * 100,
                     "평균보유일": tr.days.mean()})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> list[str]:
    fmt = {"거래수": "{:.0f}", "평균보유일": "{:.1f}", "승률": "{:.1f}%", "CAGR": "{:+.1f}%"}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) if c == cols[0] else ("-" if pd.isna(r[c]) else
                                     fmt.get(c, "{:+.1f}%").format(r[c])) for c in cols) + " |")
    return out


def main():
    bt.OUT.mkdir(exist_ok=True)
    syms = pool()
    close, vol = bt.load_prices(syms)
    spy = close.pop("SPY")
    close = close[vol.columns].dropna(axis=1, how="all")
    vol = vol[close.columns]
    print(f"후보군 {len(syms)}개 중 가격 데이터 {close.shape[1]}개")

    band = (close >= LO) & (close <= HI)
    eligible = vol.where(band & close.notna()).rank(axis=1, ascending=False, method="first") <= N
    ma_e, ma_x = close.rolling(ENTRY).mean(), close.rolling(EXIT).mean()
    up = (close.shift(1) <= ma_e.shift(1)) & (close > ma_e)
    below = close < ma_x

    # 1) 알림 그대로: 모든 신호 (종목 수 제한 없음) → 거래별 통계
    t, _ = bt.simulate(close, eligible, up, below, bt.START)
    t["ret"] = t["exit"] * (1 - COST) / (t["entry"] / (1 - COST)) - 1
    t.to_csv(bt.OUT / "ma5_20_trades.csv", index=False)
    tt = trade_table(t)

    # 2) 실제 계좌처럼: K칸 슬롯 포트폴리오
    sig = {"short": (up, below)}
    v = {"trend": False, "market": False, "rank": "day", "exit": "ma5"}  # exit=sig["short"][1] (20일선)
    s_spy = spy[spy.index >= bt.START].dropna()
    yrs = (s_spy.index[-1] - s_spy.index[0]).days / 365.25
    summ, years = [], {}
    for k in (5, 10, 20):
        pf.MAX_POS, pf.COST = k, COST
        eq, tk = pf.run(v, close, eligible, sig, pd.Series(True, index=close.index), bt.START)
        st = pf.stats(eq, tk, spy)
        tot = st.iloc[-1]
        summ.append({"방식": f"{k}칸 (한 종목 {100/k:.0f}%)", "누적수익": tot["수익률"], "CAGR": tot["CAGR"],
                     "최대낙폭": tot["MDD"], "거래수": tot["거래수"], "승률": tot["승률"]})
        years[f"{k}칸"] = st.set_index("연도")["수익률"]
    summ.append({"방식": "SPY 보유", "누적수익": (s_spy.iloc[-1] / s_spy.iloc[0] - 1) * 100,
                 "CAGR": ((s_spy.iloc[-1] / s_spy.iloc[0]) ** (1 / yrs) - 1) * 100,
                 "최대낙폭": ((s_spy / s_spy.cummax()) - 1).min() * 100, "거래수": np.nan, "승률": np.nan})
    yt = pd.DataFrame(years)
    yt["SPY"] = st.set_index("연도")["SPY"]
    yt = yt.reset_index().rename(columns={"index": "연도"})

    md = [f"# 5일선 돌파 매수 → 20일선 이탈 매도 백테스트 ({bt.START[:4]} ~ {close.index[-1].date()})", "",
          f"- 대상: 매일 주가 ${LO}~${HI} 종목 중 거래량 상위 {N}",
          f"- 후보군: 현재 거래량 상위 250 + S&P 500 이력 종목 = {close.shape[1]}개 (생존 편향 있음)",
          "- 신호일 종가 매수, 매수 다음 날부터 종가 < 20일선이면 그날 종가 매도",
          f"- 비용: 매수·매도 각 {COST*100:.1f}%", "",
          "## 계좌 수익률 (K칸 슬롯: 신호마다 총자산/K 매수, 칸이 차면 당일 상승률 순)", ""]
    md += md_table(pd.DataFrame(summ))
    md += ["", "## 연도별 계좌 수익률", ""] + md_table(yt)
    md += ["", "## 거래별 통계 (알림이 온 모든 신호)", ""] + md_table(tt)
    print(f"전체 신호 {len(t)}건")
    best = t.sort_values("ret", ascending=False).head(10)
    md += ["", "## 최고 수익 거래 Top 10", ""] + [
        f"- {r.symbol} {r.entry_date.date()} → {r.exit_date.date()} ({r.days}일) {r.ret*100:+.0f}%"
        for r in best.itertuples()]
    md += ["", "## 가장 많이 거래된 종목", ""] + [
        f"- {s}: {c}회" for s, c in t["symbol"].value_counts().head(10).items()]
    (bt.OUT / "ma5_20.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
