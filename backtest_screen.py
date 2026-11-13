#!/usr/bin/env python3
"""매도 소진 후 반등 전략 백테스트 (screen_rsi.py 규칙).

관찰: 주가 $5~$40, RSI(14)<=30, 거래량 5일 연속 감소, 거래량 < 20일 평균
매수: 관찰 조건 충족 후 10거래일 안의 첫 5일선 상향 돌파일 종가
매도: 매수 다음 날부터 종가 < 20일선 × (1 - EXIT_PCT) 인 날 종가   (기본 5%)
후보군: 현재 상장된 미국 정규 거래소 종목 (주가 $1 이상) — 상장폐지 종목 없음 (생존 편향)
비용: 매수·매도 각 0.1%
python backtest_screen.py → results/screen_bt.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import alert as al
import backtest as bt
import portfolio as pf
import screen_rsi as sr

COST, EXIT_PCT = 0.001, 0.05
W = sr.SC["watch_days"]
PLAIN = os.environ.get("PLAIN") == "1"   # 1이면 RSI·거래량 조건 없이 가격대 + 5일선 돌파만
TAG = "_plain" if PLAIN else ""


def universe() -> list[str]:
    import yfinance as yf
    from yfinance import EquityQuery as Q

    q = Q("and", [Q("eq", ["region", "us"]), Q("is-in", ["exchange", *al.CONFIG["exchanges"]]),
                  Q("gte", ["intradayprice", 1])])
    out, off = [], 0
    while off < 9000:
        res = yf.screen(q, sortField="dayvolume", sortAsc=False, size=250, offset=off)
        qs = [x["symbol"] for x in res.get("quotes", []) if x.get("symbol")]
        out += qs
        if len(qs) < 250:
            break
        off += 250
    return sorted(set(out))


def load(symbols):
    import yfinance as yf

    cs, vs = [], []
    for i in range(0, len(symbols), 500):
        chunk = symbols[i:i + 500]
        raw = yf.download(chunk, start=bt.WARMUP_START, interval="1d", group_by="ticker",
                          auto_adjust=True, threads=True, progress=False)
        lv = raw.columns.levels[0]
        c = pd.DataFrame({s: raw[s]["Close"] for s in chunk if s in lv}).dropna(axis=1, how="all")
        cs.append(c.astype("float32"))
        vs.append(pd.DataFrame({s: raw[s]["Volume"] for s in c.columns}).astype("float32"))
        print(f"  다운로드 {min(i + 500, len(symbols))}/{len(symbols)}", flush=True)
    close, vol = pd.concat(cs, axis=1), pd.concat(vs, axis=1)
    idx = pd.DatetimeIndex(close.index)
    close.index = vol.index = idx.tz_localize(None) if idx.tz is not None else idx
    return close.sort_index(), vol.sort_index()


def entries(close, vol):
    if PLAIN:
        ma5 = close.rolling(5).mean()
        band = (close >= sr.SC["min_price"]) & (close <= sr.SC["max_price"])
        return ((close.shift(1) <= ma5.shift(1)) & (close > ma5) & band).fillna(False)
    su = sr.setups(close, vol).values
    ma5 = close.rolling(5).mean()
    cross = ((close.shift(1) <= ma5.shift(1)) & (close > ma5)).values
    ent = np.zeros(su.shape, dtype=bool)
    for j in range(su.shape[1]):
        last, used = -10**9, -1
        sj, cj = su[:, j], cross[:, j]
        for i in np.flatnonzero(sj | cj):
            if sj[i]:
                last = i
            if cj[i] and i - last <= W and used != last:
                ent[i, j] = True
                used = last
    return pd.DataFrame(ent, index=close.index, columns=close.columns)


def trades(close, ent, ma20, start, armed: bool):
    """거래별 시뮬레이션. armed=True면 20일선 위로 올라선 뒤에만 매도 규칙 적용(60일 못 오르면 종료)."""
    C, E, M = close.values, ent.values, ma20.values
    i0 = close.index.searchsorted(pd.Timestamp(start))
    out = []
    for j in range(C.shape[1]):
        ei, arm = None, False
        for i in range(i0, len(C)):
            if ei is None:
                if E[i, j] and not np.isnan(C[i, j]):
                    ei, arm = i, (not armed) or C[i, j] > M[i, j]
                continue
            p = C[i, j]
            if np.isnan(p):
                out.append((j, ei, i, C[i - 1, j])); ei = None; continue
            if not arm:
                if p > M[i, j]:
                    arm = True
                elif i - ei > sr.SC["track_days"]:
                    out.append((j, ei, i, p)); ei = None
                continue
            if p < M[i, j] * (1 - EXIT_PCT):
                out.append((j, ei, i, p)); ei = None
        if ei is not None:
            out.append((j, ei, len(C) - 1, C[-1, j]))
    t = pd.DataFrame(out, columns=["j", "ei", "xi", "exit"])
    t["symbol"] = close.columns[t["j"]]
    t["entry_date"], t["exit_date"] = close.index[t["ei"]], close.index[t["xi"]]
    t["entry"] = C[t["ei"], t["j"]]
    t["days"] = t["xi"] - t["ei"]
    t["ret"] = t["exit"] * (1 - COST) / (t["entry"] / (1 - COST)) - 1
    return t.drop(columns=["j", "ei", "xi"])


def live_trades(close, ent, ma20, start):
    """현재 알림 규칙: 모든 매수 신호별로 매도일 계산.
    기준선 = 20일선 × (1-EXIT_PCT). 기준선 위로 올라온 적 있으면 기준선 아래 마감 시 매도,
    없으면(돌파 때 이미 기준선 아래) 돌파 가격 아래 마감 시 매도."""
    C, E, M = close.values, ent.values, ma20.values * (1 - EXIT_PCT)
    i0 = close.index.searchsorted(pd.Timestamp(start))
    out, n = [], len(C)
    for j in range(C.shape[1]):
        held_until = -1
        for ei in np.flatnonzero(E[i0:, j]) + i0:
            ep = C[ei, j]
            if np.isnan(ep) or ei <= held_until:   # 추적 중인 종목의 재돌파는 알림 없음
                continue
            arm, xi, xp, op = ep >= M[ei, j], n - 1, C[n - 1, j], True
            for i in range(ei + 1, n):
                p = C[i, j]
                if np.isnan(p):
                    xi, xp, op = i, C[i - 1, j], False; break
                if not arm:
                    if p >= M[i, j]:
                        arm = True
                    elif p < ep:
                        xi, xp, op = i, p, False; break
                elif p < M[i, j]:
                    xi, xp, op = i, p, False; break
            if np.isnan(xp):
                xp = ep
            out.append((j, ei, xi, ep, xp, op, bool(ep >= M[ei, j])))
            held_until = xi
    t = pd.DataFrame(out, columns=["j", "ei", "xi", "entry", "exit", "open", "above_line"])
    t["symbol"] = close.columns[t["j"]]
    t["entry_date"], t["exit_date"] = close.index[t["ei"]], close.index[t["xi"]]
    t["days"] = t["xi"] - t["ei"]
    t["ret"] = t["exit"] * (1 - COST) / (t["entry"] / (1 - COST)) - 1
    return t


def slots(t, close, k, start):
    """K칸 계좌: 신호마다 총자산/K 매수(칸이 차면 당일 상승률 순), 신호별 매도일에 매도."""
    C = close.values
    chg = close.pct_change(fill_method=None).values
    i0 = close.index.searchsorted(pd.Timestamp(start))
    by_day = {i: g for i, g in t.groupby("ei")}
    cash, pos, eq, taken = 1.0, {}, [], []
    val = lambda i, j: C[i, j] if not np.isnan(C[i, j]) else pos[j][3]
    for i in range(i0, len(C)):
        for j in [j for j, p in pos.items() if p[1] == i and not p[2]]:
            cash += pos[j][0] * pos[j][4] * (1 - COST); del pos[j]
        for j in pos:
            if not np.isnan(C[i, j]):
                pos[j][3] = C[i, j]
        if i in by_day and len(pos) < k:
            total = cash + sum(p[0] * val(i, j) for j, p in pos.items())
            g = by_day[i]; g = g[~g["j"].isin(pos)]
            g = g.assign(c=chg[i, g["j"]]).sort_values("c", ascending=False)
            for r in g.head(k - len(pos)).itertuples():
                amt = min(total / k, cash)
                if amt <= 0:
                    break
                pos[r.j] = [amt * (1 - COST) / r.entry, r.xi, r.open, r.entry, r.exit]
                cash -= amt; taken.append(r.Index)
        eq.append(cash + sum(p[0] * val(i, j) for j, p in pos.items()))
    return pd.Series(eq, index=close.index[i0:]), t.loc[taken]


def ttable(t):
    rows = []
    for y, tr in list(t.groupby(t["entry_date"].dt.year)) + [("전체", t)]:
        rows.append({"연도": y, "거래수": len(tr), "승률": (tr.ret > 0).mean() * 100,
                     "평균수익": tr.ret.mean() * 100, "중간값": tr.ret.median() * 100,
                     "평균이익": tr.ret[tr.ret > 0].mean() * 100, "평균손실": tr.ret[tr.ret <= 0].mean() * 100,
                     "평균보유일": tr.days.mean()})
    return pd.DataFrame(rows)


def mdt(df):
    fmt = {"거래수": "{:.0f}", "평균보유일": "{:.1f}", "승률": "{:.1f}%"}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) if c == cols[0] else ("-" if pd.isna(r[c]) else
                   fmt.get(c, "{:+.1f}%").format(r[c])) for c in cols) + " |")
    return out


def main():
    bt.OUT.mkdir(exist_ok=True)
    syms = universe()
    print(f"후보군 {len(syms)}종목")
    close, vol = load(syms + ["SPY"])
    spy = close.pop("SPY").astype(float)
    vol = vol.drop(columns="SPY")
    print(f"가격 데이터 {close.shape[1]}종목, {close.index[0].date()} ~ {close.index[-1].date()}")

    ent = entries(close, vol)
    ma20 = close.rolling(20).mean()
    below = close < ma20 * (1 - EXIT_PCT)
    n_sig = int(ent[ent.index >= bt.START].values.sum())
    print(f"매수 신호 {n_sig}건")

    t1 = live_trades(close, ent, ma20, bt.START)
    t1.drop(columns=["j", "ei", "xi"]).to_csv(bt.OUT / f"screen_bt{TAG}_trades.csv", index=False)
    s_spy = spy[spy.index >= bt.START].dropna()
    yrs = (s_spy.index[-1] - s_spy.index[0]).days / 365.25
    summ, years = [], {}
    for k in (5, 10, 20):
        eq, tk = slots(t1, close.astype(float), k, bt.START)
        st = pf.stats(eq, tk, spy)
        tot = st.iloc[-1]
        eq.to_csv(bt.OUT / f"screen_bt{TAG}_equity_{k}.csv")
        summ.append({"방식": f"{k}칸 (한 종목 {100/k:.0f}%)", "누적수익": tot["수익률"], "CAGR": tot["CAGR"],
                     "최대낙폭": tot["MDD"], "거래수": tot["거래수"], "승률": tot["승률"]})
        years[f"{k}칸"] = st.set_index("연도")["수익률"]
    summ.append({"방식": "SPY 보유", "누적수익": (s_spy.iloc[-1] / s_spy.iloc[0] - 1) * 100,
                 "CAGR": ((s_spy.iloc[-1] / s_spy.iloc[0]) ** (1 / yrs) - 1) * 100,
                 "최대낙폭": ((s_spy / s_spy.cummax()) - 1).min() * 100, "거래수": np.nan, "승률": np.nan})
    yt = pd.DataFrame(years); yt["SPY"] = st.set_index("연도")["SPY"]
    yt = yt.reset_index().rename(columns={"index": "연도"})

    lo, hi = t1[~t1.above_line], t1[t1.above_line]
    md = [f"# 매도 소진 후 5일선 돌파 백테스트 — 현재 알림 규칙 ({bt.START[:4]} ~ {close.index[-1].date()})", "",
          *(["- 조건: 주가 $5~$40 (RSI·거래량 조건 없음)", "- 매수: 5일선 상향 돌파일 종가"] if PLAIN else
            ["- 관찰: 주가 $5~$40, RSI(14)≤30, 거래량 5일 연속 감소, 거래량 < 20일 평균",
             f"- 매수: 조건 충족 후 {W}거래일 안 첫 5일선 돌파일 종가"]),
          f"- 매도: 종가 < 20일선 × {1-EXIT_PCT:.2f} (기준선). 돌파 때 이미 기준선 아래였으면, 기준선 위로 "
          "올라오기 전에는 돌파 가격 아래로 마감 시 매도",
          f"- 후보군: 현재 상장 종목 {close.shape[1]}개 (상장폐지 종목 없음 → 생존 편향), 비용 매수·매도 각 0.1%", "",
          "## 계좌 수익률 (K칸: 신호마다 총자산/K 매수, 칸이 차면 당일 상승률 순)", ""]
    md += mdt(pd.DataFrame(summ)) + ["", "## 연도별 계좌 수익률", ""] + mdt(yt)
    md += ["", "## 거래별 통계 (모든 신호)", "",
           f"매수 다음 날 바로 매도: {(t1.days == 1).mean()*100:.1f}% · "
           f"돌파 때 기준선 아래였던 신호: {len(lo)/len(t1)*100:.0f}% "
           f"(승률 {(lo.ret>0).mean()*100:.0f}%, 평균 {lo.ret.mean()*100:+.1f}%) · "
           f"기준선 위였던 신호: 승률 {(hi.ret>0).mean()*100:.0f}%, 평균 {hi.ret.mean()*100:+.1f}%", ""]
    md += mdt(ttable(t1))
    per_day = t1.groupby("ei").size()
    md += ["", f"하루 평균 신호 {len(t1) / (len(close.index[close.index >= bt.START])):.0f}건 (최대 {per_day.max()}건)"]
    best, worst = t1.nlargest(8, "ret"), t1.nsmallest(8, "ret")
    md += ["", "## 최고/최저 거래", ""]
    md += [f"- {r.symbol} {r.entry_date.date()} → {r.exit_date.date()} ({r.days}일) {r.ret*100:+.0f}%"
           for r in pd.concat([best, worst]).itertuples()]
    (bt.OUT / f"screen_bt{TAG}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
