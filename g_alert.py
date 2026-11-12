#!/usr/bin/env python3
"""G 전략 일일 리포트 → 텔레그램.

규칙 (백테스트 G와 동일)
- 대상: S&P 500 구성 종목 중 주가 $5 이상 $40 이하, 그날 거래량 상위 100
- 매수: 40주선 상향 돌파 (지난주 종가 <= 40주선, 현재가 > 40주선) · 6개월 상승률 높은 순
- 매도: 현재가 < 40주선 (매수 다음 날부터)
- 최대 5종목, 각 총자산의 20%

장 마감 20분 전(15:40 ET 전후)에 하루 한 번 실행.
  python g_alert.py              # 실행 시간대(15:30~16:30 ET)에만, 하루 1회
  python g_alert.py --force      # 시간·중복 무시하고 실행 (수동 실행용)
  python g_alert.py --dry-run    # 텔레그램 전송·상태 저장 없이 화면 출력
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, time as dtime

import pandas as pd
import requests

import alert as al
import backtest as bt

MAX_POS, MIN_PRICE, MAX_PRICE, TOP_N = 5, 5, 40, 100  # 주가 $5 이상 $40 이하
STATE = al.ROOT / "state_g.json"
SEED = al.ROOT / "results" / "g_positions.json"
SP500_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
             "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def in_window(now: datetime) -> bool:
    return now.weekday() < 5 and dtime(15, 30) <= now.time() <= dtime(16, 30)


def sp500_members() -> list[str]:
    try:
        r = requests.get(SP500_URL, timeout=30)
        r.raise_for_status()
        h = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"[경고] S&P 500 목록 최신본 실패, 저장본 사용: {e}", file=sys.stderr)
        h = pd.read_csv(al.ROOT / "data" / "sp500_history.csv")
    last = h.sort_values("date").iloc[-1]
    print(f"S&P 500 구성 기준일 {last['date']}")
    return [t.replace(".", "-") for t in last["tickers"].split(",")]


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    st = {"positions": {}, "last_run": "", "log": []}
    if SEED.exists():  # 처음 실행: 백테스트의 현재 보유 종목으로 시작
        seed = json.loads(SEED.read_text(encoding="utf-8"))
        st["positions"] = seed["positions"]
        st["seeded_from"] = seed["as_of"]
        print(f"백테스트 보유 종목으로 시작 ({seed['as_of']} 기준): {', '.join(seed['positions']) or '없음'}")
    return st


def download(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(symbols, period="2y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    lv = raw.columns.levels[0]
    close = pd.DataFrame({s: raw[s]["Close"] for s in symbols if s in lv})
    vol = pd.DataFrame({s: raw[s]["Volume"] for s in symbols if s in lv})
    idx = pd.DatetimeIndex(close.index)
    close.index = vol.index = idx.tz_localize(None) if idx.tz is not None else idx
    close = close.dropna(how="all")
    return close, vol.reindex(close.index)


def decide(close, vol, members, positions, today_str):
    """오늘의 매도·매수 결정. 반환: sells, buys, info(종목별 지표)."""
    sig = bt.signals(close)
    up40, below40, ma40 = sig["long"][0].iloc[-1], sig["long"][1].iloc[-1], sig["ma40"].iloc[-1]
    px = close.iloc[-1]
    mom6 = close.iloc[-1] / close.iloc[-127] - 1 if len(close) > 127 else px * float("nan")
    chg = close.pct_change(fill_method=None).iloc[-1]
    info = {s: {"price": float(px[s]), "ma40": float(ma40[s]), "mom6": float(mom6[s]),
                "chg": float(chg[s])} for s in close.columns if pd.notna(px[s])}

    sells = []
    for s, p in positions.items():
        if s in info and today_str > p["since"] and bool(below40.get(s, False)):
            sells.append(s)
    remaining = len(positions) - len(sells)

    mem = [s for s in members if s in info]
    band = [s for s in mem if MIN_PRICE <= info[s]["price"] <= MAX_PRICE and pd.notna(vol[s].iloc[-1])]
    top = set(vol.iloc[-1][band].sort_values(ascending=False).index[:TOP_N])
    cands = [s for s in top if bool(up40.get(s, False)) and s not in positions and pd.notna(info[s]["mom6"])]
    cands.sort(key=lambda s: info[s]["mom6"], reverse=True)
    buys = cands[:max(0, MAX_POS - remaining)]
    return sells, buys, cands, info


def fmt_date(d: str) -> str:
    return d[5:].replace("-", "/")


def message(now, today, sells, buys, waiting, positions_before, positions_after, info, meta) -> str:
    L = [f"📊 G전략 리포트 {fmt_date(today)} (뉴욕 {now:%H:%M})", ""]
    if buys:
        L.append(f"🟢 매수 {len(buys)}종목 (각 총자산의 20%)")
        for s in buys:
            i = info[s]
            L += [f"• {al.label(s, meta)}",
                  f"   ${i['price']:.2f} ({i['chg']*100:+.1f}%)  6개월 {i['mom6']*100:+.0f}%  40주선 ${i['ma40']:.2f}"]
        L.append("")
    if sells:
        L.append(f"🔴 매도 {len(sells)}종목 (40주선 이탈)")
        for s in sells:
            p, i = positions_before[s], info[s]
            L += [f"• {al.label(s, meta)}",
                  f"   ${i['price']:.2f}  매수 {fmt_date(p['since'])} ${p['price']:.2f} → "
                  f"{(i['price']/p['price']-1)*100:+.1f}%"]
        L.append("")
    if not buys and not sells:
        L += ["오늘 매매 없음", ""]
    L.append(f"📋 보유 {len(positions_after)}/{MAX_POS}")
    for s, p in sorted(positions_after.items(), key=lambda x: x[1]["since"]):
        i = info.get(s)
        if not i:
            L.append(f"• {s} (가격 정보 없음)")
            continue
        L += [f"• {al.label(s, meta)}",
              f"   {fmt_date(p['since'])} ${p['price']:.2f} → ${i['price']:.2f} ({(i['price']/p['price']-1)*100:+.1f}%)"
              f"  40주선 대비 {(i['price']/i['ma40']-1)*100:+.1f}%"]
    if waiting:
        L += ["", "⏳ 자리가 없어 대기 중인 돌파 종목: " + ", ".join(waiting[:5])]
    L += ["", "※ 장 마감 직전 가격 기준이라 종가와 조금 다를 수 있어요.",
          "※ 백테스트 기반 참고용이며 투자 판단은 본인 책임이에요."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    force = args.force or os.environ.get("FORCE") == "1"
    now = datetime.now(al.NY)
    st = load_state()
    if not force and (not in_window(now) or st.get("last_run") == now.strftime("%Y-%m-%d")):
        print(f"실행 시간이 아니거나 오늘 이미 실행함 ({now:%a %H:%M} ET) → 건너뜀")
        return 0

    members = sp500_members()
    positions = st["positions"]
    close, vol = download(sorted(set(members) | set(positions)))
    today = close.index[-1].strftime("%Y-%m-%d")
    print(f"가격 데이터 {close.shape[1]}종목, 기준일 {today}")

    sells, buys, cands, info = decide(close, vol, members, positions, today)
    before = dict(positions)
    after = {s: p for s, p in positions.items() if s not in sells}
    for s in buys:
        after[s] = {"since": today, "price": round(info[s]["price"], 4)}
    waiting = [s for s in cands if s not in buys]
    print(f"매도 {sells}, 매수 {buys}, 대기 {waiting}")

    meta = al.load_meta()
    al.fill_meta(meta, sorted(set(after) | set(sells)), {})
    msg = message(now, today, sells, buys, waiting, before, after, info, meta)
    if args.dry_run:
        print("\n" + msg)
        return 0
    al.send_telegram(msg)
    st["positions"] = after
    st["last_run"] = now.strftime("%Y-%m-%d")
    st["log"] = (st.get("log", []) + [{"date": today, "buy": buys, "sell": sells}])[-500:]
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    al.META_PATH.write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
