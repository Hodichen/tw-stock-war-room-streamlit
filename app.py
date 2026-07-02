from __future__ import annotations

import json
import os
import requests
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

APP_TITLE = "台股 8:50 盤前戰情室"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root{
  --bg:#0d0a05; --panel:#151009; --panel2:#20170d; --gold:#d4af37;
  --gold2:#f3d37a; --cream:#f5e6c8; --muted:#b8aa8a; --line:rgba(212,175,55,.24);
  --red:#d36d5c; --green:#83d6a2; --blue:#8ab4f8; --orange:#f6b26b;
}
html, body, .stApp{background:radial-gradient(circle at 15% 0%, #2d2211 0%, #0d0a05 38%, #070503 100%); color:var(--cream);} 
.block-container{padding-top:1.2rem; padding-bottom:4rem; max-width:1480px;}
section[data-testid="stSidebar"]{display:none!important;} div[data-testid="collapsedControl"]{display:none!important;}
h1,h2,h3{color:var(--cream); letter-spacing:.01em;} p, li{line-height:1.7;}
.hero{border:1px solid rgba(212,175,55,.35); background:linear-gradient(135deg, rgba(212,175,55,.15), rgba(16,12,8,.92)); border-radius:28px; padding:1.45rem 1.65rem; box-shadow:0 16px 48px rgba(0,0,0,.35); margin:.4rem 0 1rem;}
.hero-title{font-size:2.3rem; font-weight:900; color:var(--cream);}
.hero-sub{color:var(--muted); margin-top:.35rem;}
.version-box{border:1px solid rgba(212,175,55,.32); background:rgba(212,175,55,.07); border-radius:20px; padding:.9rem 1rem; margin:.2rem 0 1rem;}
.verdict{border-left:7px solid var(--red); background:rgba(211,109,92,.16); border-radius:18px; padding:1.05rem 1.15rem; margin:1rem 0; font-size:1.15rem; font-weight:760; color:var(--cream);} 
.verdict.bull{border-left-color:var(--green); background:rgba(131,214,162,.13);} .verdict.neutral{border-left-color:var(--gold); background:rgba(212,175,55,.12);} 
.kpi{border:1px solid var(--line); background:linear-gradient(180deg, rgba(32,23,13,.85), rgba(13,10,5,.88)); border-radius:22px; padding:1rem 1.1rem; min-height:118px;}
.kpi-label{color:var(--muted); font-size:.9rem;}.kpi-value{color:var(--cream); font-size:1.85rem; font-weight:850; margin-top:.45rem}.kpi-note{color:var(--muted); font-size:.86rem; margin-top:.2rem;}
.module{border:1px solid var(--line); background:linear-gradient(180deg, rgba(28,20,10,.96), rgba(12,9,5,.96)); border-radius:22px; padding:1.05rem 1.12rem; min-height:205px; box-shadow:0 12px 34px rgba(0,0,0,.22);}
.module-top{display:flex; align-items:center; justify-content:space-between; gap:.6rem; margin-bottom:.45rem}.module-title{color:var(--gold2); font-size:1.05rem; font-weight:850}.module-chip{border:1px solid rgba(212,175,55,.35); border-radius:999px; color:var(--gold2); padding:.2rem .55rem; font-size:.82rem; background:rgba(212,175,55,.08)}
.module-main{font-size:1.55rem; color:var(--cream); font-weight:900; margin:.35rem 0}.module-small{color:var(--muted); font-size:.92rem; line-height:1.62}.evidence{margin-top:.7rem; padding-top:.65rem; border-top:1px solid rgba(212,175,55,.16); color:var(--cream); font-size:.93rem;}
.card{border:1px solid var(--line); background:rgba(18,14,8,.92); border-radius:22px; padding:1.05rem 1.15rem; margin-bottom:.85rem;}
.card h4{color:var(--gold2); margin:0 0 .4rem; font-size:1.05rem}.card p{margin:.2rem 0; color:var(--cream)}
.ai-box{border:1px solid rgba(138,180,248,.35); background:rgba(138,180,248,.08); border-radius:22px; padding:1.1rem 1.2rem; margin:1rem 0;}
.ai-box h3{margin-top:0}.warn{border:1px solid rgba(211,109,92,.35); background:rgba(211,109,92,.1); border-radius:16px; padding:.75rem .85rem; color:var(--cream);}
.stDataFrame{border-radius:18px; overflow:hidden}.stTabs [data-baseweb="tab-list"]{gap:.35rem}.stTabs [data-baseweb="tab"]{background:rgba(22,16,8,.85); color:var(--cream); border:1px solid rgba(212,175,55,.22); border-radius:999px; padding:.55rem 1.1rem}.stTabs [aria-selected="true"]{background:rgba(212,175,55,.22)!important; border-color:rgba(212,175,55,.55)!important;}
div[role="radiogroup"]{gap:.8rem;} div[role="radiogroup"] label{border:1px solid rgba(212,175,55,.35); border-radius:18px; padding:.75rem 1.05rem; background:rgba(212,175,55,.08); min-width:210px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@dataclass
class DataBundle:
    etf: pd.DataFrame
    risk: pd.DataFrame
    forum: pd.DataFrame
    market: pd.DataFrame


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_data() -> DataBundle:
    return DataBundle(
        etf=read_csv_safely(DATA_DIR / "etf_holdings_sample.csv"),
        risk=read_csv_safely(DATA_DIR / "risk_stocks_sample.csv"),
        forum=read_csv_safely(DATA_DIR / "forum_mentions_sample.csv"),
        market=read_csv_safely(DATA_DIR / "market_snapshot_sample.csv"),
    )


def normalize_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def bias_score(market: pd.DataFrame, names: list[str]) -> tuple[int, str, str]:
    if market.empty or "item" not in market.columns:
        return 0, "資料不足", "尚未有市場資料"
    m = market.copy()
    m["item"] = m["item"].astype(str)
    m = normalize_numeric(m, ["value"])
    subset = m[m["item"].isin(names)]
    if subset.empty:
        return 0, "資料不足", "尚未有對應資料"
    signed = []
    evidence = []
    for _, row in subset.iterrows():
        bias = str(row.get("bias", "")).lower()
        value = float(row.get("value", 0))
        if bias == "bearish" or value < 0:
            signed.append(-1)
        elif bias == "bullish" or value > 0:
            signed.append(1)
        else:
            signed.append(0)
        evidence.append(f"{row.get('item')} {value:g}：{row.get('note', '')}")
    score = int(sum(signed))
    direction = "偏多" if score > 0 else "偏空" if score < 0 else "中性"
    return score, direction, "｜".join(evidence[:3])


def score_stocks(etf: pd.DataFrame, risk: pd.DataFrame, forum: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol", "name", "sector", "etf_count", "avg_weight", "weight_3d_change", "weight_5d_change",
        "weight_10d_change", "shares_change_10d", "etf_score", "forum_score", "forum_mentions",
        "risk_penalty", "total_score", "conclusion", "why"
    ]
    if etf.empty:
        return pd.DataFrame(columns=columns)
    etf = etf.copy()
    for col in ["symbol", "name", "sector", "etf_code"]:
        if col in etf.columns:
            etf[col] = etf[col].astype(str)
    etf = normalize_numeric(etf, ["weight", "weight_3d_change", "weight_5d_change", "weight_10d_change", "shares_change_10d"])
    grouped = etf.groupby(["symbol", "name", "sector"], as_index=False).agg(
        etf_count=("etf_code", "nunique"), avg_weight=("weight", "mean"),
        weight_3d_change=("weight_3d_change", "mean"), weight_5d_change=("weight_5d_change", "mean"),
        weight_10d_change=("weight_10d_change", "mean"), shares_change_10d=("shares_change_10d", "sum"),
    )
    grouped["etf_score"] = (
        grouped["etf_count"].clip(0, 5) * 10 + grouped["weight_3d_change"].clip(-1, 1) * 22
        + grouped["weight_5d_change"].clip(-1, 1) * 20 + grouped["weight_10d_change"].clip(-1, 1) * 18
        + (grouped["shares_change_10d"] > 0).astype(int) * 8
    ).clip(0, 100)
    if not forum.empty and "symbol" in forum.columns:
        forum = forum.copy(); forum["symbol"] = forum["symbol"].astype(str); forum = normalize_numeric(forum, ["mentions"])
        fs = forum.groupby("symbol", as_index=False).agg(forum_mentions=("mentions", "sum"))
        mx = max(float(fs["forum_mentions"].max()), 1.0)
        fs["forum_score"] = (fs["forum_mentions"] / mx * 100).round(1)
        grouped = grouped.merge(fs, on="symbol", how="left")
    else:
        grouped["forum_score"] = 0; grouped["forum_mentions"] = 0
    grouped[["forum_score", "forum_mentions"]] = grouped[["forum_score", "forum_mentions"]].fillna(0)
    if not risk.empty and "symbol" in risk.columns:
        risk = risk.copy(); risk["symbol"] = risk["symbol"].astype(str)
        penalty_map = {"high": 30, "medium": 15, "low": 5}
        risk["risk_penalty"] = risk.get("risk_level", "low").astype(str).map(penalty_map).fillna(5)
        rp = risk.groupby("symbol", as_index=False).agg(risk_penalty=("risk_penalty", "max"))
        grouped = grouped.merge(rp, on="symbol", how="left")
    else:
        grouped["risk_penalty"] = 0
    grouped["risk_penalty"] = grouped["risk_penalty"].fillna(0)
    grouped["total_score"] = (grouped["etf_score"] * .58 + grouped["forum_score"] * .15 + grouped["avg_weight"].clip(0, 8) * 2.2 - grouped["risk_penalty"]).round(1).clip(0, 100)
    grouped["conclusion"] = grouped.apply(lambda r: "風險優先，不追" if r.risk_penalty >= 25 else "強勢觀察" if r.total_score >= 78 else "觀察池" if r.total_score >= 68 else "題材追蹤" if r.total_score >= 56 else "僅追蹤", axis=1)
    grouped["why"] = grouped.apply(lambda r: "；".join([x for x in [
        f"{int(r.etf_count)}檔ETF共識" if r.etf_count >= 2 else "",
        f"3日權重+{r.weight_3d_change:.2f}" if r.weight_3d_change > 0 else "",
        f"10日權重+{r.weight_10d_change:.2f}" if r.weight_10d_change > 0 else "",
        f"論壇{int(r.forum_mentions)}次" if r.forum_mentions > 0 else "",
        f"風險扣{int(r.risk_penalty)}" if r.risk_penalty > 0 else "",
    ] if x][:4]) or "資料不足", axis=1)
    return grouped.sort_values(["total_score", "etf_count"], ascending=False).reset_index(drop=True)


def sector_scores(stocks: pd.DataFrame) -> pd.DataFrame:
    if stocks.empty:
        return pd.DataFrame(columns=["sector", "sector_score", "stocks", "count"])
    out = stocks.groupby("sector", as_index=False).agg(
        sector_score=("total_score", "mean"), count=("symbol", "count"),
        stocks=("symbol", lambda x: "、".join(map(str, list(x)[:5]))),
    )
    out["sector_score"] = out["sector_score"].round(1)
    return out.sort_values("sector_score", ascending=False)


def round_price(x: float) -> float:
    if x >= 1000: step = 5
    elif x >= 500: step = 1
    elif x >= 100: step = 0.5
    elif x >= 50: step = 0.1
    else: step = 0.05
    return round(round(x / step) * step, 2)


def secret_get(name: str, default=None):
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    if value is None:
        value = os.getenv(name, default)
    return value


def parse_price_payload(payload) -> dict[str, float]:
    """支援常見 API 池格式：
    1. {"2330": 1415, "2408": 76.2}
    2. {"data": [{"symbol":"2330", "price":1415}]}
    3. [{"symbol":"2330", "close":1415}]
    """
    out: dict[str, float] = {}
    rows = payload
    if isinstance(payload, dict):
        if "data" in payload:
            rows = payload["data"]
        elif "result" in payload:
            rows = payload["result"]
        else:
            # dict mapping: symbol -> price 或 symbol -> object
            for k, v in payload.items():
                if isinstance(v, (int, float, str)):
                    try:
                        out[str(k)] = float(str(v).replace(',', ''))
                    except Exception:
                        pass
                elif isinstance(v, dict):
                    for key in ["price", "last", "close", "current", "reference_price", "ref_price"]:
                        if key in v:
                            try:
                                out[str(k)] = float(str(v[key]).replace(',', ''))
                                break
                            except Exception:
                                pass
            return out
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or row.get("stock_id") or row.get("code") or row.get("ticker") or "").strip()
            if not symbol:
                continue
            for key in ["price", "last", "close", "current", "reference_price", "ref_price", "成交價", "收盤價"]:
                if key in row:
                    try:
                        out[symbol] = float(str(row[key]).replace(',', ''))
                        break
                    except Exception:
                        pass
    return out


@st.cache_data(ttl=60)
def fetch_price_pool(symbols: tuple[str, ...]) -> tuple[dict[str, float], str]:
    """從你的 API 池抓股價。沒設定時回傳空 dict，使用示範價。"""
    csv_url = secret_get("PUBLIC_PRICE_CSV_URL") or secret_get("GOOGLE_STOCK_CSV_URL")
    api_url = secret_get("PUBLIC_PRICE_API_URL") or secret_get("GOOGLE_STOCK_API_URL")
    api_key = secret_get("PUBLIC_PRICE_API_KEY") or secret_get("GOOGLE_STOCK_API_KEY")
    method = str(secret_get("PUBLIC_PRICE_API_METHOD", "GET")).upper()
    symbols_csv = ",".join(symbols)

    # 最簡單：Google Sheet 發布成 CSV，欄位支援 symbol, price / close / last
    if csv_url:
        try:
            df = pd.read_csv(csv_url)
            payload = df.to_dict("records")
            return parse_price_payload(payload), "Google Sheet CSV"
        except Exception as exc:
            return {}, f"CSV API 失敗：{exc}"

    if not api_url:
        return {}, "未設定 API 池，使用示範價"

    try:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["x-api-key"] = api_key
        if method == "POST":
            resp = requests.post(api_url, headers=headers, json={"symbols": list(symbols)}, timeout=12)
        else:
            resp = requests.get(api_url, headers=headers, params={"symbols": symbols_csv}, timeout=12)
        resp.raise_for_status()
        return parse_price_payload(resp.json()), f"API 池：{method}"
    except Exception as exc:
        return {}, f"API 池失敗，使用示範價：{exc}"


def add_trade_plan(stocks: pd.DataFrame, market_label: str) -> pd.DataFrame:
    if stocks.empty:
        return stocks
    # 正式版優先讀你的 API 池；抓不到才用示範價。
    symbols = tuple(stocks["symbol"].astype(str).tolist())
    api_prices, api_note = fetch_price_pool(symbols)
    demo_prices = {
        "2330": 1415.0, "3017": 748.0, "2408": 76.2, "6669": 3150.0, "3661": 2920.0,
        "2382": 307.0, "2454": 1320.0, "8299": 675.0, "6274": 295.0, "4966": 775.0,
        "2356": 64.5, "3443": 1210.0,
    }
    merged_prices = {**demo_prices, **api_prices}
    out = stocks.copy()
    out["reference_price"] = out["symbol"].astype(str).map(merged_prices).fillna(100.0)
    bearish = "偏空" in market_label or "觀望" in market_label
    if bearish:
        out["entry_price"] = out["reference_price"].apply(lambda p: round_price(p * 1.018))
        out["stop_price"] = out["reference_price"].apply(lambda p: round_price(p * 0.985))
        out["target_price"] = out["reference_price"].apply(lambda p: round_price(p * 1.04))
        out["entry_condition"] = "只做轉強價：9:10後站上進場價且不能跌回；大盤未翻多不追高"
    else:
        out["entry_price"] = out["reference_price"].apply(lambda p: round_price(p * 1.008))
        out["stop_price"] = out["reference_price"].apply(lambda p: round_price(p * 0.98))
        out["target_price"] = out["reference_price"].apply(lambda p: round_price(p * 1.045))
        out["entry_condition"] = "站上進場價且量能放大；跌回昨收附近不追"
    out["price_note"] = api_note
    return out

def market_verdict(us_score: int, fut_score: int, mode: str) -> tuple[str, str, str]:
    # 防呆：美股與台指夜盤同空，不允許輸出偏多。
    if us_score < 0 and fut_score < 0:
        return "偏空 / 觀望", "隔夜風向與台指夜盤同向偏空，今天不應輸出『上漲』結論；先防守，不追高。", "bear"
    if us_score > 0 and fut_score > 0:
        if mode.startswith("8:50"):
            return "偏多但需確認", "美股與夜盤偏多，但8:50仍要等開盤後量價確認，避免假開高。", "bull"
        return "偏多 / 可觀察強勢延續", "9:10後若權值與主線同步站穩，可提高觀察股權重。", "bull"
    return "中性 / 等待確認", "外部風向與夜盤不同步，先看9:10開盤確認，不用急著判斷方向。", "neutral"


def module_summary(bundle: DataBundle, stocks: pd.DataFrame) -> dict:
    us_score, us_dir, us_evd = bias_score(bundle.market, ["道瓊", "S&P 500", "Nasdaq", "費半", "VIX"])
    fut_score, fut_dir, fut_evd = bias_score(bundle.market, ["台指期夜盤"])
    top_sector = sector_scores(stocks).iloc[0].to_dict() if not sector_scores(stocks).empty else {"sector":"資料不足", "sector_score":0, "stocks":""}
    risk_count = len(bundle.risk) if not bundle.risk.empty else 0
    high_risk = int((bundle.risk.get("risk_level", pd.Series(dtype=str)).astype(str) == "high").sum()) if not bundle.risk.empty else 0
    forum_mentions = int(pd.to_numeric(bundle.forum.get("mentions", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not bundle.forum.empty else 0
    return {
        "us_score": us_score, "us_dir": us_dir, "us_evidence": us_evd,
        "fut_score": fut_score, "fut_dir": fut_dir, "fut_evidence": fut_evd,
        "top_sector": top_sector,
        "risk_count": risk_count, "high_risk": high_risk,
        "forum_mentions": forum_mentions,
    }


def fallback_ai(verdict: str, ms: dict, watch: pd.DataFrame) -> str:
    top_names = "、".join((watch["symbol"].astype(str) + " " + watch["name"].astype(str)).head(3).tolist()) if not watch.empty else "暫無"
    if "偏空" in verdict:
        return f"AI輔助摘要：今天主策略不是追價，而是找抗跌與轉強確認。美股/費半與台指夜盤偏空時，ETF熱區只能當中期觀察池。優先觀察 {top_names} 是否在9:10後站上條件價；未站上就不做。"
    if "偏多" in verdict:
        return f"AI輔助摘要：外部風向支持偏多，但仍要看9:10後量價是否延續。優先追蹤ETF共識度高且沒有處置風險的 {top_names}，跌回失效價則降級。"
    return f"AI輔助摘要：目前訊號分歧，今天應以確認為主。{top_names} 可列入觀察，但必須等大盤與個股同時轉強，不用預設方向。"


def call_ai(verdict: str, ms: dict, module_rows: list[dict], watch: pd.DataFrame) -> str:
    provider = str(secret_get("AI_PROVIDER", "google")).lower()
    payload = {
        "market_verdict": verdict,
        "modules": module_rows,
        "watchlist": watch[["symbol", "name", "sector", "total_score", "entry_price", "stop_price", "entry_condition"]].head(5).to_dict("records") if not watch.empty else [],
    }
    prompt = """
你是台股盤前交易研究助理。請根據客觀證據輸出繁體中文摘要。
規則：
1. 若美股與台指期夜盤同空，不能寫偏多，最多只能寫觀望/防守。
2. ETF只代表中期資金熱區，不等於今日方向。
3. KOL/論壇只代表注意力，不代表勝率。
4. 每檔股票必須寫條件價、失效價、為何觀察。
5. 不要使用保證語氣，不要說必漲。
請輸出：簡單結論、今日主線、三檔觀察股、風險提醒、失效條件。
資料：
""" + json.dumps(payload, ensure_ascii=False)

    # 預設使用 Google Gemini。適合你說的 Google API 池。
    if provider in ["google", "gemini"]:
        key = secret_get("GOOGLE_API_KEY") or secret_get("GEMINI_API_KEY")
        if not key:
            return fallback_ai(verdict, ms, watch) + "\n\n（尚未設定 GOOGLE_API_KEY / GEMINI_API_KEY，暫用規則摘要。）"
        try:
            from google import genai
            client = genai.Client(api_key=key)
            model = secret_get("GOOGLE_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
            resp = client.models.generate_content(model=model, contents=prompt)
            return getattr(resp, "text", str(resp))
        except Exception as exc:
            return fallback_ai(verdict, ms, watch) + f"\n\n（Gemini API 呼叫失敗，已使用規則摘要。錯誤：{exc}）"

    # 保留 OpenAI fallback，之後要切回也不用重寫。
    key = secret_get("OPENAI_API_KEY")
    if not key:
        return fallback_ai(verdict, ms, watch)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = secret_get("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        resp = client.responses.create(model=model, input=prompt, max_output_tokens=700)
        return getattr(resp, "output_text", str(resp))
    except Exception as exc:
        return fallback_ai(verdict, ms, watch) + f"\n\n（OpenAI API 呼叫失敗，已使用規則摘要。錯誤：{exc}）"

def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>", unsafe_allow_html=True)


def render_module(no: int, title: str, main: str, direction: str, score: str, evidence: str) -> None:
    st.markdown(f"""
    <div class='module'>
      <div class='module-top'><div class='module-title'>{no}. {title}</div><div class='module-chip'>{score}</div></div>
      <div class='module-main'>{main}</div>
      <div class='module-small'>方向：{direction}</div>
      <div class='evidence'>證據：{evidence}</div>
    </div>
    """, unsafe_allow_html=True)


bundle = load_data()
stocks_raw = score_stocks(bundle.etf, bundle.risk, bundle.forum)
ms = module_summary(bundle, stocks_raw)

st.markdown(f"""
<div class='hero'>
  <div class='hero-title'>{APP_TITLE}</div>
  <div class='hero-sub'>先看客觀證據，再由 AI 輔助統整。ETF 是資金熱區，不是今日方向；KOL 是注意力，不是勝率。</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='version-box'><b>報告版本切換</b><br><span style='color:#b8aa8a'>8:50 = 盤前劇本；9:10 = 開盤確認。左側上傳功能已移除，正式資料改走 API / CSV 自動更新。</span></div>", unsafe_allow_html=True)
mode = st.radio("報告版本", ["8:50 盤前版", "9:10 開盤確認版"], horizontal=True, label_visibility="collapsed", index=1)

verdict, simple_reason, verdict_type = market_verdict(ms["us_score"], ms["fut_score"], mode)
stocks = add_trade_plan(stocks_raw, verdict)
watch = stocks[(stocks["risk_penalty"] < 25) & (stocks["total_score"] >= 50)].head(8)

vclass = "bull" if verdict_type == "bull" else "neutral" if verdict_type == "neutral" else ""
st.markdown(f"<div class='verdict {vclass}'>簡單結論：{verdict}。{simple_reason}</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: kpi("美股 / 費半分數", str(ms["us_score"]), ms["us_dir"])
with c2: kpi("台指期夜盤分數", str(ms["fut_score"]), ms["fut_dir"])
with c3: kpi("ETF 熱區", str(ms["top_sector"].get("sector", "-")), f"分數 {ms['top_sector'].get('sector_score', 0)}")
with c4: kpi("風險股票", str(ms["risk_count"]), f"高風險 {ms['high_risk']} 檔")
with c5: kpi("論壇 / KOL 熱度", str(ms["forum_mentions"]), "注意力，不等於勝率")

st.markdown("## 五個模組的客觀證據")
module_rows = [
    {"module":"美股 / 費半", "main":ms["us_dir"], "direction":ms["us_dir"], "score":ms["us_score"], "evidence":ms["us_evidence"]},
    {"module":"台指期夜盤", "main":ms["fut_dir"], "direction":ms["fut_dir"], "score":ms["fut_score"], "evidence":ms["fut_evidence"]},
    {"module":"主動 ETF", "main":str(ms["top_sector"].get("sector", "資料不足")), "direction":"中期熱區，不決定今日方向", "score":ms["top_sector"].get("sector_score", 0), "evidence":f"代表股：{ms['top_sector'].get('stocks', '')}"},
    {"module":"處置 / 注意股", "main":f"{ms['risk_count']} 檔風險股", "direction":"風控扣分", "score":-ms["high_risk"], "evidence":f"高風險 {ms['high_risk']} 檔；處置、注意、解除都不等於利多"},
    {"module":"論壇 / KOL", "main":"注意力集中" if ms["forum_mentions"] else "資料不足", "direction":"注意力，不等於勝率", "score":ms["forum_mentions"], "evidence":"熱門討論只用來判斷是否過熱，不作為買進理由"},
]
cols = st.columns(5)
for idx, row in enumerate(module_rows, start=1):
    with cols[idx-1]:
        render_module(idx, row["module"], str(row["main"]), str(row["direction"]), f"分數 {row['score']}", str(row["evidence"]))

st.markdown("## 今日觀察股與明確條件價")
st.markdown("<div class='warn'>價位說明：目前是示範價位。正式串接即時行情 API 後，會自動改成昨收、現價、9:10 五分鐘高點、停損價。偏空日只給『轉強價』，沒有站上就不進。</div>", unsafe_allow_html=True)
if not watch.empty:
    display_cols = ["symbol", "name", "sector", "total_score", "conclusion", "reference_price", "entry_price", "stop_price", "target_price", "entry_condition", "why"]
    st.dataframe(
        watch[display_cols].rename(columns={
            "symbol":"代號", "name":"股票", "sector":"產業", "total_score":"總分", "conclusion":"等級",
            "reference_price":"參考價", "entry_price":"條件進場價", "stop_price":"失效/停損價", "target_price":"第一目標", "entry_condition":"進場條件", "why":"原因"
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("目前沒有達到條件的觀察股。")

st.markdown("## AI 輔助統整")
ai_text = call_ai(verdict, ms, module_rows, watch)
st.markdown(f"<div class='ai-box'><h3>AI 綜合結論</h3><p>{ai_text.replace(chr(10), '<br>')}</p></div>", unsafe_allow_html=True)

st.markdown("## 模組細節")
t1, t2, t3, t4, t5, t6 = st.tabs(["總覽", "主動ETF熱區", "美股/夜盤", "處置注意股", "論壇KOL", "API設定"])
with t1:
    st.dataframe(pd.DataFrame(module_rows), use_container_width=True, hide_index=True)
with t2:
    left, right = st.columns([1,1])
    with left:
        st.markdown("### 產業熱區")
        st.dataframe(sector_scores(stocks), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### ETF共識股")
        st.dataframe(stocks.head(12), use_container_width=True, hide_index=True)
with t3:
    st.dataframe(bundle.market, use_container_width=True, hide_index=True)
with t4:
    st.dataframe(bundle.risk, use_container_width=True, hide_index=True)
with t5:
    st.dataframe(bundle.forum, use_container_width=True, hide_index=True)
with t6:
    st.markdown("""
### 最方便的串接順序
1. **先接行情價位 API**：讓條件價、停損價、9:10 五分鐘高點變成真實數字。
2. **再接市場證據 API**：美股、費半、台指期夜盤。
3. **最後接主動 ETF 持股與論壇/KOL**：因為格式最不穩定，先做可替換 adapter。

### AI API：先用 Google / Gemini
- Streamlit Cloud 請到 **Manage app → Settings → Secrets** 放 API key，不要寫進 GitHub。
- Secrets 範例：
```toml
AI_PROVIDER = "google"
GOOGLE_API_KEY = "你的 Google / Gemini API key"
GOOGLE_MODEL = "gemini-2.5-flash"
```
- 若沒有設定 API key，畫面會自動使用規則版 AI 摘要，不會掛掉。

### 你的公開資訊股價 API 池
支援兩種最方便格式：
```toml
# A. Google Sheet 發布成 CSV
PUBLIC_PRICE_CSV_URL = "https://docs.google.com/spreadsheets/d/e/.../pub?output=csv"

# B. Google Apps Script / 其他 API endpoint
PUBLIC_PRICE_API_URL = "https://script.google.com/macros/s/.../exec"
PUBLIC_PRICE_API_METHOD = "GET"
PUBLIC_PRICE_API_KEY = "如果有key才填"
```
欄位至少要有 symbol/code/stock_id 其中之一，價格欄位支援 price/last/close/current/reference_price。
""")

st.caption(f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}｜本系統為研究儀表板，不是投資建議。")
