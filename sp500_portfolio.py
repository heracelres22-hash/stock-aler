#!/usr/bin/env python3
"""S&P 500 (시점별 실제 구성 종목) 대상 소수 종목 포트폴리오 백테스트.

대상: 그날 S&P 500 구성 종목 중 주가 MIN_PRICE 이상 MAX_PRICE 미만, 그중 거래량 상위 100.
구성 종목 이력: data/sp500_history.csv (fja05680/sp500). 마지막 기록 이후는 마지막 구성 유지.
규칙·비용·방식 A~G는 portfolio.py와 동일.
python sp500_portfolio.py → results/sp500_portfolio.md
"""
from __future__ import annotations

import json

import pandas as pd

import backtest as bt
import portfolio as pf

MIN_PRICE, MAX_PRICE = 20, 50


def membership() -> pd.DataFrame:
    h = pd.read_csv(bt.ROOT / "data" / "sp500_history.csv", parse_dates=["date"])
    rows = {r.date: {t.replace(".", "-") for t in r.tickers.split(",")} for r in h.itertuples()}
    syms = sorted(set().union(*rows.values()))
    return pd.DataFrame({d: {s: s in m for s in syms} for d, m in rows.items()}).T.sort_index()


def main():
    bt.OUT.mkdir(exist_ok=True)
    mem = membership()
    syms = list(mem.columns)
    close, vol = bt.load_prices(syms)
    spy = close.pop("SPY")
    close = close[vol.columns].dropna(axis=1, how="all")
    vol = vol[close.columns]
    missing = sorted(set(syms) - set(close.columns))
    print(f"S&P 500 이력 종목 {len(syms)}개 중 가격 데이터 {close.shape[1]}개 (없음 {len(missing)}개)")

    member = mem.reindex(close.index, method="ffill").reindex(columns=close.columns).fillna(False).astype(bool)
    px_ok = (close >= MIN_PRICE) & (close < MAX_PRICE)
    rank = vol.where(member & px_ok & close.notna()).rank(axis=1, ascending=False, method="first")
    eligible = rank <= bt.CFG["top_n"]
    print("하루 평균 매수 대상 종목 수:", round(eligible[eligible.index >= bt.START].sum(axis=1).mean(), 1))

    sig = bt.signals(close)
    spy_ok = spy > bt.signals(spy.to_frame())["ma40"]["SPY"]
    s_spy = spy[spy.index >= bt.START].dropna()
    yrs = (s_spy.index[-1] - s_spy.index[0]).days / 365.25

    md = [f"# S&P 500 백테스트 (최대 {pf.MAX_POS}종목, {bt.START[:4]} ~ {close.index[-1].date()})", "",
          f"- 대상: 그날 S&P 500 구성 종목 중 주가 ${MIN_PRICE} 이상 ${MAX_PRICE} 미만, 거래량 상위 {bt.CFG['top_n']}",
          "- 구성 종목은 시점별 실제 이력 사용 (상장폐지·인수된 종목 중 Yahoo에 데이터 없는 "
          f"{len(missing)}개는 제외)",
          f"- 각 종목 = 총자산의 1/{pf.MAX_POS}, 신호일 종가 체결, 비용 매수·매도 각 {pf.COST*100:.1f}%",
          "- 주가 조건은 매수 시점에만 적용 (보유 중 $50 넘어도 계속 보유)", "", "## 요약", "",
          "| 방식 | 설명 | 누적수익 | 연평균(CAGR) | 최대낙폭 | 거래수 | 승률 | 평균보유일 |",
          "|---|---|---|---|---|---|---|---|"]
    years, top = {}, []
    for k, v in pf.VARIANTS.items():
        eq, t = pf.run(v, close, eligible, sig, spy_ok, bt.START)
        t.to_csv(bt.OUT / f"sp500_trades_{k}.csv", index=False)
        if k == "G":  # 실시간 알림 초기 보유 종목
            op = t[t["open"]]
            (bt.OUT / "g_positions.json").write_text(json.dumps(
                {"as_of": str(close.index[-1].date()),
                 "positions": {r.symbol: {"since": str(r.entry_date.date()), "price": round(float(r.entry), 4)}
                               for r in op.itertuples()}}, indent=1), encoding="utf-8")
        st = pf.stats(eq, t, spy)
        tot = st.iloc[-1]
        md.append(f"| {k} | {v['desc']} | {tot['수익률']:+.0f}% | {tot['CAGR']:+.1f}% | {tot['MDD']:.1f}% | "
                  f"{tot['거래수']:.0f} | {tot['승률']:.1f}% | {tot['평균보유일']:.1f} |")
        years[k] = st.set_index("연도")["수익률"]
        best = t.sort_values("ret", ascending=False).head(5)
        top.append(f"- {k}: " + ", ".join(f"{r.symbol} {r.ret*100:+.0f}%" for r in best.itertuples()))
    md.append(f"| SPY | 보유만 | {(s_spy.iloc[-1]/s_spy.iloc[0]-1)*100:+.0f}% | "
              f"{((s_spy.iloc[-1]/s_spy.iloc[0])**(1/yrs)-1)*100:+.1f}% | "
              f"{((s_spy/s_spy.cummax())-1).min()*100:.1f}% | - | - | - |")
    yt = pd.DataFrame(years)
    yt["SPY"] = pf.stats(eq, t, spy).set_index("연도")["SPY"]
    md += ["", "## 연도별 수익률", "", "| 연도 | " + " | ".join(yt.columns) + " |",
           "|" + "---|" * (len(yt.columns) + 1)]
    md += [f"| {y} | " + " | ".join(f"{x:+.1f}%" for x in r) + " |" for y, r in yt.iterrows()]
    md += ["", "## 방식별 최고 수익 거래 Top 5", ""] + top
    md += ["", f"데이터 없는 종목: {', '.join(missing)}"]
    (bt.OUT / "sp500_portfolio.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
