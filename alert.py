#!/usr/bin/env python3
"""미국 거래량 상위 종목 이동평균 상향 돌파 → 텔레그램 알림.

사용법:
  python alert.py              # 장중일 때만 확인 후 알림 전송
  python alert.py --force      # 장 시간과 관계없이 실행
  python alert.py --dry-run    # 텔레그램 대신 화면에 출력 (상태 저장 안 함)
  python alert.py --test       # 텔레그램 연결 테스트 메시지 전송
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
STATE_PATH = ROOT / "state.json"
META_PATH = ROOT / "meta.json"          # 종목 영문명·섹터 캐시 (자동 생성)
KO_NAMES = json.loads((ROOT / "ko_names.json").read_text(encoding="utf-8"))
SECTOR_KO = {
    "Technology": "기술", "Financial Services": "금융", "Healthcare": "헬스케어",
    "Consumer Cyclical": "경기소비재", "Consumer Defensive": "필수소비재",
    "Communication Services": "통신·미디어", "Industrials": "산업재", "Energy": "에너지",
    "Utilities": "유틸리티", "Real Estate": "부동산", "Basic Materials": "소재",
}
NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- 시장 시간
def market_is_open(now: datetime) -> bool:
    """정규장(09:30~16:00 ET) + 종가 반영용 15분 여유. 공휴일은 데이터가 안 바뀌어 중복 알림이 안 감."""
    if now.weekday() >= 5:
        return False
    return dtime(9, 30) <= now.time() <= dtime(16, 15)


# ---------------------------------------------------------------- 종목 선정
def top_by_volume(n: int, exchanges: list[str], min_price: float) -> tuple[list[str], dict[str, str]]:
    """미국 정규 거래소 상장, 주가 min_price 이상 종목을 당일 거래량 순으로 (시총 조건 없음)."""
    import yfinance as yf
    from yfinance import EquityQuery as Q

    query = Q("and", [
        Q("eq", ["region", "us"]),
        Q("is-in", ["exchange", *exchanges]),
        Q("gte", ["intradayprice", min_price]),
    ])
    res = yf.screen(query, sortField="dayvolume", sortAsc=False, size=n)
    quotes = [q for q in res.get("quotes", []) if q.get("symbol")][:n]
    if not quotes:
        raise RuntimeError("거래량 상위 종목을 가져오지 못했습니다.")
    names = {q["symbol"]: q.get("longName") or q.get("shortName") or "" for q in quotes}
    return [q["symbol"] for q in quotes], names


# ---------------------------------------------------------------- 종목 정보
def load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def fill_meta(meta: dict, symbols: list[str], screen_names: dict[str, str]) -> None:
    """캐시에 없는 종목만 Yahoo에서 영문명·섹터 조회."""
    import yfinance as yf

    for sym in symbols:
        if sym in meta and meta[sym].get("sector"):
            continue
        name, sector = screen_names.get(sym, ""), ""
        try:
            info = yf.Ticker(sym).info
            name = info.get("longName") or info.get("shortName") or name
            sector = info.get("sector") or ("ETF" if info.get("quoteType") == "ETF" else "")
        except Exception as e:  # 실패해도 알림은 보냄
            print(f"[경고] {sym} 정보 조회 실패: {e}", file=sys.stderr)
        meta[sym] = {"name": name, "sector": sector}
        if sym not in KO_NAMES:
            print(f"[참고] 한글명 없음: {sym} ({name}) → ko_names.json에 추가 가능")


def label(sym: str, meta: dict) -> str:
    m = meta.get(sym, {})
    en, ko = re.sub(r",?\s+(Inc\.?|Corp\.?|Corporation|Co\.|Ltd\.?|plc|N\.V\.|S\.A\.|Holdings?,? Inc\.?)$", "",
                    m.get("name", "")).strip(), KO_NAMES.get(sym, "")
    sector = SECTOR_KO.get(m.get("sector", ""), m.get("sector", ""))
    name = f"{ko} ({en})" if ko and en else (ko or en)
    return f"{sym} {name}".strip() + (f" · {sector}" if sector else "")


def download_daily(symbols: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        symbols, period="2y", interval="1d", group_by="ticker",
        auto_adjust=True, threads=True, progress=False,
    )
    out = {}
    for s in symbols:
        try:
            df = raw[s].dropna(subset=["Close"])
        except KeyError:
            continue
        if not df.empty:
            out[s] = df
    return out


# ---------------------------------------------------------------- 신호 판정
def crossed_up(series: pd.Series, window: int) -> tuple[bool, float | None]:
    """직전 봉: 종가 <= 이평, 현재 봉: 종가 > 이평 이면 상향 돌파."""
    series = series.dropna()
    if len(series) < window + 1:
        return False, None
    ma = series.rolling(window).mean()
    prev_c, prev_ma = series.iloc[-2], ma.iloc[-2]
    cur_c, cur_ma = series.iloc[-1], ma.iloc[-1]
    return bool(prev_c <= prev_ma and cur_c > cur_ma), float(cur_ma)


def closes(daily: pd.DataFrame) -> pd.Series:
    close = daily["Close"].astype(float).dropna()
    idx = pd.DatetimeIndex(close.index)
    close.index = idx.tz_localize(None) if idx.tz is not None else idx
    return close


def check_exit(daily: pd.DataFrame, kind: str, cfg: dict, since: str) -> dict | None:
    """돌파 알림 후 감시 중인 종목이 (돌파 다음 날 이후) 현재가 < 이평 이면 이탈.
    kind="short": 일봉 5일선, kind="long": 주봉 40주선(진행 중인 주는 현재가)."""
    close = closes(daily)
    if len(close) < 2:
        return None
    last_day = close.index[-1].strftime("%Y-%m-%d")
    if last_day <= since:  # 돌파 당일의 장중 되돌림은 다음 날 판단
        return None
    if kind == "short":
        series, window = close, cfg["short_ma_days"]
    else:
        series, window = close.resample("W-FRI").last().dropna(), cfg["long_ma_weeks"]
    if len(series) < window:
        return None
    ma = float(series.rolling(window).mean().iloc[-1])
    price = float(close.iloc[-1])
    if price >= ma:
        return None
    return {"kind": f"exit_{kind}", "ma": ma, "price": price,
            "chg": (price / float(close.iloc[-2]) - 1) * 100, "period": last_day}


def detect(daily: pd.DataFrame, short_days: int, long_weeks: int) -> list[dict]:
    close = closes(daily)
    price = float(close.iloc[-1])
    chg = (price / float(close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0.0
    last_day = close.index[-1]

    found = []
    ok, ma = crossed_up(close, short_days)
    if ok:
        found.append({"kind": "short", "ma": ma, "price": price, "chg": chg,
                      "period": last_day.strftime("%Y-%m-%d")})

    # 주봉: 금요일 기준 주간 마지막 종가 (진행 중인 주는 현재가)
    weekly = close.resample("W-FRI").last().dropna()
    ok, ma = crossed_up(weekly, long_weeks)
    if ok:  # 중복 방지는 하루 단위 (같은 주라도 매일 다시 알림)
        found.append({"kind": "long", "ma": ma, "price": price, "chg": chg,
                      "period": last_day.strftime("%Y-%m-%d")})
    return found


# ---------------------------------------------------------------- 상태(중복 방지)
def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            st.setdefault("sent", {})
            act = st.setdefault("active", {})
            if act and not set(act) <= {"short", "long"}:  # 예전 형식(5일선만) 변환
                act = st["active"] = {"short": act}
            act.setdefault("short", {})
            act.setdefault("long", {})
            return st
        except json.JSONDecodeError:
            pass
    return {"sent": {}, "active": {"short": {}, "long": {}}}


def save_state(state: dict, now: datetime) -> None:
    cutoff = (now - timedelta(days=21)).isoformat()
    state["sent"] = {k: v for k, v in state["sent"].items() if v >= cutoff}
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------- 텔레그램
def chat_ids() -> list[str]:
    return [c.strip() for c in os.environ.get("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    ids = chat_ids()
    if not token or not ids:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_IDS 환경변수가 필요합니다.")
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):  # 텔레그램 4096자 제한
        if len(buf) + len(line) > 3800:
            chunks.append(buf)
            buf = ""
        buf += line
    chunks.append(buf)
    for cid in ids:
        for chunk in chunks:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": cid, "text": chunk,
                                    "disable_web_page_preview": "true"}, timeout=20)
            if not r.ok:
                print(f"[경고] {cid} 전송 실패: {r.status_code} {r.text}", file=sys.stderr)


def format_message(hits: dict[str, list[tuple[str, dict]]], now: datetime, cfg: dict,
                   meta: dict | None = None) -> str:
    meta = meta or {}
    titles = {"short": f"🟢 {cfg['short_ma_days']}일선 상향 돌파",
              "long": f"🔵 {cfg['long_ma_weeks']}주선 상향 돌파",
              "exit_long": f"🟠 {cfg['long_ma_weeks']}주선 하향 이탈 (돌파 알림 후)",
              "exit_short": f"🔴 {cfg['short_ma_days']}일선 하향 이탈 (돌파 알림 후)"}
    lines = [f"📈 이동평균 신호 알림  {now:%m/%d %H:%M} (뉴욕)",
             f"대상: 미국 거래량 상위 {cfg['top_n']}종목 (주가 ${cfg['min_price']:g} 이상)", ""]
    for kind in ("long", "short", "exit_long", "exit_short"):
        if not hits.get(kind):
            continue
        lines.append(titles[kind])
        for sym, h in sorted(hits[kind], key=lambda x: x[1]["chg"], reverse=True):  # 상승률 순
            gap = (h["price"] / h["ma"] - 1) * 100
            lines.append(f"• {label(sym, meta)}")
            lines.append(f"   ${h['price']:.2f} ({h['chg']:+.1f}%)  이평 ${h['ma']:.2f} 대비 {gap:+.1f}%")
            if kind.startswith("exit"):
                ret = (h["price"] / h["entry_price"] - 1) * 100
                lines.append(f"   돌파 {h['since'][5:].replace('-', '/')} ${h['entry_price']:.2f} → 수익률 {ret:+.1f}%")
        lines.append("")
    if cfg.get("intraday_note", True):
        lines.append("※ 장중 신호는 종가 전까지 되돌려질 수 있어요.")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------- 메인
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    force = args.force or os.environ.get("FORCE") == "1"

    now = datetime.now(NY)
    if args.test:
        send_telegram(f"✅ 주식 알림 봇 연결 성공 ({now:%Y-%m-%d %H:%M} 뉴욕)")
        print("테스트 메시지 전송 완료")
        return 0
    if not force and not market_is_open(now):
        print(f"장 시간이 아님 ({now:%a %H:%M} ET) → 건너뜀")
        return 0

    cfg = CONFIG
    symbols, screen_names = top_by_volume(cfg["top_n"], cfg["exchanges"], cfg["min_price"])
    state = load_state()
    # 돌파 후 보유 중인 종목은 상위 100에서 빠져도 이탈 여부를 계속 확인
    act = state["active"]
    held = set(act["short"]) | set(act["long"])
    watch = symbols + sorted(held - set(symbols))
    data = download_daily(watch)
    print(f"종목 {len(watch)}개 중 {len(data)}개 데이터 확보 "
          f"(이탈 감시 5일선 {len(act['short'])}, 40주선 {len(act['long'])})")

    hits: dict[str, list] = {"short": [], "long": [], "exit_short": [], "exit_long": []}

    # 1) 이탈 확인 (돌파 알림을 받았던 종목만)
    for kind in ("short", "long"):
        for sym, pos in list(act[kind].items()):
            df = data.get(sym)
            if df is None:
                continue
            h = check_exit(df, kind, cfg, pos["since"])
            if h:
                h.update(since=pos["since"], entry_price=pos["price"])
                hits[h["kind"]].append((sym, h))
                del act[kind][sym]

    # 2) 돌파 확인
    for sym in symbols:  # 거래량 순서 유지
        df = data.get(sym)
        if df is None:
            continue
        for h in detect(df, cfg["short_ma_days"], cfg["long_ma_weeks"]):
            if sym not in act[h["kind"]]:
                act[h["kind"]][sym] = {"since": h["period"], "price": round(h["price"], 4)}
            key = f"{sym}:{h['kind']}:{h['period']}"
            if key in state["sent"]:
                continue
            hits[h["kind"]].append((sym, h))
            state["sent"][key] = now.isoformat()

    total = sum(len(v) for v in hits.values())
    print(f"새 신호 {total}건 (5일선 {len(hits['short'])}, 40주선 {len(hits['long'])}, "
          f"5일선 이탈 {len(hits['exit_short'])}, 40주선 이탈 {len(hits['exit_long'])})")
    if total == 0:
        if not args.dry_run:
            save_state(state, now)
        return 0

    meta = load_meta()
    fill_meta(meta, [s for k in hits for s, _ in hits[k]], screen_names)
    if not args.dry_run:
        META_PATH.write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    msg = format_message(hits, now, cfg, meta)
    if args.dry_run:
        print("\n" + msg)
        return 0
    send_telegram(msg)
    save_state(state, now)
    return 0


if __name__ == "__main__":
    sys.exit(main())
