#!/usr/bin/env python3
"""알림 기준 백테스트: 이평선 상향 돌파 시 매수, 다시 이평선 아래로 내려가면 매도.

- 전략 A: 5일선 돌파 매수 → 종가 < 5일선 매도
- 전략 B: 40주선 돌파 매수 → 종가 < 40주선 매도 (주봉, 진행 중인 주는 당일 종가)
- 대상: 매일 그날 거래량 상위 N종목 (종가 >= min_price). 후보군은 현재 Yahoo 거래량 상위 250.
- 체결: 신호 발생일 종가 매수, 이탈일 종가 매도. 수수료·세금·슬리피지 미반영.
- 포트폴리오: 보유 중인 종목들에 매일 동일 비중 (보유 종목 없으면 현금 0%).

python backtest.py  → results/backtest.md, results/trades_*.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
OUT = ROOT / "results"
START, WARMUP_START = "2021-01-01", "2020-06-01"
POOL_SIZE = 250


def candidate_pool() -> list[str]:
    import yfinance as yf
    from yfinance import EquityQuery as Q

    q = Q("and", [Q("eq", ["region", "us"]), Q("is-in", ["exchange", *CFG["exchanges"]]),
                  Q("gte", ["intradayprice", CFG["min_price"]])])
    res = yf.screen(q, sortField="dayvolume", sortAsc=False, size=POOL_SIZE)
    return [x["symbol"] for x in res.get("quotes", []) if x.get("symbol")]


def load_prices(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(symbols + ["SPY"], start=WARMUP_START, interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    close = pd.DataFrame({s: raw[s]["Close"] for s in symbols + ["SPY"] if s in raw.columns.levels[0]})
    vol = pd.DataFrame({s: raw[s]["Volume"] for s in symbols if s in raw.columns.levels[0]})
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = vol.index = idx
    return close.sort_index(), vol.sort_index()


def week_label(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx + pd.to_timedelta((4 - idx.weekday) % 7, unit="D")  # 해당 주 금요일


def signals(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n, w = CFG["short_ma_days"], CFG["long_ma_weeks"]
    ma5 = close.rolling(n).mean()
    up5 = (close.shift(1) <= ma5.shift(1)) & (close > ma5)
    below5 = close < ma5

    weekly = close.resample("W-FRI").last()
    wk = week_label(close.index)
    prev_sum = weekly.rolling(w - 1, min_periods=w - 1).sum().shift(1).reindex(wk)
    prev_wc = weekly.shift(1).reindex(wk)
    prev_wma = weekly.rolling(w, min_periods=w).mean().shift(1).reindex(wk)
    for df in (prev_sum, prev_wc, prev_wma):
        df.index = close.index
    ma40 = (prev_sum + close) / w  # 진행 중인 주 = 당일 종가
    up40 = (prev_wc <= prev_wma) & (close > ma40)
    below40 = close < ma40
    return {"short": (up5, below5), "long": (up40, below40), "ma5": ma5, "ma40": ma40}


def simulate(close, eligible, up, below, start) -> tuple[pd.DataFrame, pd.DataFrame]:
    """종목별 진입·청산 시뮬레이션. 반환: 거래 목록, 보유 매트릭스(당일 수익을 받는지)."""
    dates = close.index
    hv = np.zeros(close.shape, dtype=bool)
    trades = []
    start_i = dates.searchsorted(pd.Timestamp(start))
    c, e, u, b = close.values, eligible.values, up.fillna(False).values, below.fillna(False).values
    for j, sym in enumerate(close.columns):
        entry_i = None
        for i in range(start_i, len(dates)):
            if entry_i is None:
                if u[i, j] and e[i, j] and not np.isnan(c[i, j]):
                    entry_i = i
            else:
                hv[i, j] = True  # 진입 다음 날부터 수익 반영
                if b[i, j] or np.isnan(c[i, j]):
                    px = c[i, j] if not np.isnan(c[i, j]) else c[i - 1, j]
                    trades.append((sym, dates[entry_i], c[entry_i, j], dates[i], px, i - entry_i, False))
                    entry_i = None
        if entry_i is not None:  # 아직 보유 중
            last = len(dates) - 1
            trades.append((sym, dates[entry_i], c[entry_i, j], dates[last], c[last, j], last - entry_i, True))
    held = pd.DataFrame(hv, index=dates, columns=close.columns)
    t = pd.DataFrame(trades, columns=["symbol", "entry_date", "entry", "exit_date", "exit", "days", "open"])
    t["ret"] = t["exit"] / t["entry"] - 1
    return t, held


def yearly(trades, held, close, spy) -> pd.DataFrame:
    rets = close.pct_change(fill_method=None).where(held)
    port = rets.mean(axis=1).fillna(0.0)  # 동일 비중, 무보유일 0
    n_held = held.sum(axis=1)
    rows = []
    for y in sorted(set(port.index.year)):
        if y < int(START[:4]):
            continue
        p = port[port.index.year == y]
        eq = (1 + p).cumprod()
        mdd = (eq / eq.cummax() - 1).min()
        tr = trades[trades["entry_date"].dt.year == y]
        s = spy[spy.index.year == y].dropna()
        rows.append({
            "연도": y, "거래수": len(tr),
            "승률": (tr["ret"] > 0).mean() * 100 if len(tr) else np.nan,
            "평균수익": tr["ret"].mean() * 100 if len(tr) else np.nan,
            "평균이익": tr.loc[tr["ret"] > 0, "ret"].mean() * 100 if len(tr) else np.nan,
            "평균손실": tr.loc[tr["ret"] <= 0, "ret"].mean() * 100 if len(tr) else np.nan,
            "평균보유일": tr["days"].mean() if len(tr) else np.nan,
            "평균보유종목": n_held[n_held.index.year == y].mean(),
            "전략수익률": (eq.iloc[-1] - 1) * 100, "MDD": mdd * 100,
            "SPY": (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else np.nan,
        })
    df = pd.DataFrame(rows)
    p = port[port.index >= START]
    eq = (1 + p).cumprod()
    s = spy[spy.index >= START].dropna()
    total = {"연도": "전체", "거래수": len(trades), "승률": (trades["ret"] > 0).mean() * 100,
             "평균수익": trades["ret"].mean() * 100,
             "평균이익": trades.loc[trades["ret"] > 0, "ret"].mean() * 100,
             "평균손실": trades.loc[trades["ret"] <= 0, "ret"].mean() * 100,
             "평균보유일": trades["days"].mean(), "평균보유종목": n_held[n_held.index >= START].mean(),
             "전략수익률": (eq.iloc[-1] - 1) * 100, "MDD": ((eq / eq.cummax() - 1).min()) * 100,
             "SPY": (s.iloc[-1] / s.iloc[0] - 1) * 100}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = {"거래수": "{:.0f}", "평균보유일": "{:.1f}", "평균보유종목": "{:.1f}"}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "연도":
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("-")
            else:
                cells.append(fmt.get(c, "{:+.1f}%" if c not in ("승률",) else "{:.1f}%").format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    pool = candidate_pool()
    print(f"후보군 {len(pool)}종목")
    close, vol = load_prices(pool)
    spy = close.pop("SPY")
    close = close[vol.columns]
    print(f"가격 데이터 {close.shape[1]}종목, {close.index[0].date()} ~ {close.index[-1].date()}")

    ok_px = close >= CFG["min_price"]
    rank = vol.where(ok_px & close.notna()).rank(axis=1, ascending=False, method="first")
    eligible = rank <= CFG["top_n"]

    sig = signals(close)
    names = {"short": f"{CFG['short_ma_days']}일선 돌파 매수 → {CFG['short_ma_days']}일선 이탈 매도",
             "long": f"{CFG['long_ma_weeks']}주선 돌파 매수 → {CFG['long_ma_weeks']}주선 이탈 매도"}
    md = [f"# 백테스트 결과 ({START[:4]}년 ~ {close.index[-1].date()})", "",
          f"- 대상: 매일 그날 거래량 상위 {CFG['top_n']}종목 (주가 ${CFG['min_price']} 이상), "
          f"후보군 = 현재 거래량 상위 {len(pool)}종목",
          "- 체결: 신호일 종가 매수 / 이탈일 종가 매도, 수수료·세금 미반영",
          "- 전략수익률: 보유 종목 동일 비중, 보유 종목 없는 날은 현금",
          "- 거래 통계(거래수·승률·평균수익)는 매수한 연도 기준", ""]
    summary = {}
    for k in ("short", "long"):
        up, below = sig[k]
        trades, held = simulate(close, eligible, up, below, START)
        trades.to_csv(OUT / f"trades_{k}.csv", index=False)
        table = yearly(trades, held, close, spy)
        summary[k] = table.to_dict(orient="records")
        md += [f"## {names[k]}", "", to_md(table), ""]
        print(names[k]); print(table.to_string())
    (OUT / "backtest.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "backtest.json").write_text(json.dumps(summary, ensure_ascii=False, default=str, indent=1),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
