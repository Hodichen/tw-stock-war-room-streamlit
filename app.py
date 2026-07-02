from __future__ import annotations

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
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root {
  --bg: #0d0a05;
  --bg2: #17130d;
  --card: rgba(22, 18, 12, .94);
  --card2: rgba(35, 28, 18, .78);
  --gold: #d4af37;
  --gold2: #f3d37a;
  --cream: #f5e6c8;
  --muted: #b8aa8a;
  --line: rgba(212,175,55,.22);
  --danger: #d36d5c;
  --green: #81c995;
  --blue: #8ab4f8;
}
.stApp {
  background: radial-gradient(circle at 20% 0%, #2a2113 0%, #0d0a05 36%, #080604 100%);
  color: var(--cream);
}
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1420px; }
section[data-testid="stSidebar"] { background: #f3f4f7; }
section[data-testid="stSidebar"] * { color: #1f2937 !important; }
h1, h2, h3 { color: var(--cream); letter-spacing: .015em; }
[data-testid="stMetric"] {
  border: 1px solid var(--line);
  background: rgba(23,19,13,.76);
  border-radius: 18px;
  padding: .9rem 1rem;
  min-height: 112px;
}
[data-testid="stMetricValue"] { color: var(--cream); font-size: 1.85rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
.hero {
  border: 1px solid rgba(212,175,55,.32);
  background: linear-gradient(135deg, rgba(212,175,55,.16), rgba(23,19,13,.72));
  border-radius: 24px;
  padding: 1.35rem 1.55rem;
  box-shadow: 0 18px 50px rgba(0,0,0,.26);
  margin-bottom: 1rem;
}
.hero-title { font-size: 2.2rem; font-weight: 850; color: var(--cream); margin-bottom: .35rem; }
.hero-sub { color: var(--muted); font-size: .95rem; line-height: 1.65; }
.verdict {
  border-left: 6px solid var(--danger);
  background: rgba(211,109,92,.14);
  padding: 1rem 1.1rem;
  border-radius: 16px;
  color: var(--cream);
  margin: .75rem 0 1rem 0;
  font-size: 1.05rem;
  line-height: 1.75;
}
.verdict.bull { border-left-color: var(--green); background: rgba(129,201,149,.12); }
.verdict.neutral { border-left-color: var(--gold); background: rgba(212,175,55,.11); }
.card {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  box-shadow: 0 12px 35px rgba(0,0,0,.22);
  min-height: 132px;
  margin-bottom: .72rem;
}
.card h4 { color: var(--gold2); margin: 0 0 .55rem 0; font-size: 1.05rem; }
.card p { color: var(--cream); margin: .25rem 0; line-height: 1.7; }
.evidence-card {
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(38,29,18,.86), rgba(16,12,8,.90));
  border-radius: 18px;
  padding: .95rem 1rem;
  min-height: 138px;
}
.evidence-card .tag { color: var(--gold); font-size:.86rem; font-weight: 750; }
.evidence-card .big { color: var(--cream); font-size: 1.28rem; font-weight: 850; margin:.38rem 0; }
.evidence-card .small { color: var(--muted); line-height:1.55; font-size:.91rem; }
.badge {
  display:inline-block; padding:.23rem .58rem; border-radius:999px;
  border:1px solid rgba(212,175,55,.38); color:var(--gold2); font-size:.82rem;
  background: rgba(212,175,55,.07); margin-right:.35rem; margin-bottom:.35rem;
}
.small-muted { color: var(--muted); font-size:.88rem; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; }
.stTabs [data-baseweb="tab"] {
  background: rgba(23,19,13,.75);
  border-radius: 999px;
  color: var(--cream);
  border: 1px solid rgba(212,175,55,.18);
  padding: .5rem 1rem;
}
.stTabs [aria-selected="true"] {
  background: rgba(212,175,55,.20) !important;
  border-color: rgba(212,175,55,.55) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
def load_sample_data() -> DataBundle:
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


def upload_or_sample(label: str, sample: pd.DataFrame, key: str) -> pd.DataFrame:
    file = st.sidebar.file_uploader(label, type=["csv"], key=key)
    if file is not None:
        try:
            return pd.read_csv(file)
        except Exception as exc:
            st.sidebar.error(f"CSV 讀取失敗：{exc}")
    return sample


def score_stocks(etf: pd.DataFrame, risk: pd.DataFrame, forum: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol", "name", "sector", "etf_count", "avg_weight", "weight_3d_change",
        "weight_5d_change", "weight_10d_change", "shares_change_10d", "etf_score",
        "forum_score", "forum_mentions", "risk_penalty", "total_score", "conclusion", "why"
    ]
    if etf.empty:
        return pd.DataFrame(columns=columns)

    etf = etf.copy()
    for col in ["symbol", "name", "sector", "etf_code"]:
        if col in etf.columns:
            etf[col] = etf[col].astype(str)
    etf = normalize_numeric(etf, [
        "weight", "weight_3d_change", "weight_5d_change", "weight_10d_change", "shares_change_10d"
    ])

    grouped = (
        etf.groupby(["symbol", "name", "sector"], as_index=False)
        .agg(
            etf_count=("etf_code", "nunique"),
            avg_weight=("weight", "mean"),
            weight_3d_change=("weight_3d_change", "mean"),
            weight_5d_change=("weight_5d_change", "mean"),
            weight_10d_change=("weight_10d_change", "mean"),
            shares_change_10d=("shares_change_10d", "sum"),
        )
    )

    grouped["etf_score"] = (
        grouped["etf_count"].clip(0, 5) * 10
        + grouped["weight_3d_change"].clip(-1, 1) * 22
        + grouped["weight_5d_change"].clip(-1, 1) * 20
        + grouped["weight_10d_change"].clip(-1, 1) * 18
        + (grouped["shares_change_10d"] > 0).astype(int) * 8
    ).clip(0, 100)

    if not forum.empty and "symbol" in forum.columns:
        forum = forum.copy()
        forum["symbol"] = forum["symbol"].astype(str)
        forum = normalize_numeric(forum, ["mentions"])
        forum_score = forum.groupby("symbol", as_index=False).agg(forum_mentions=("mentions", "sum"))
        max_mentions = max(float(forum_score["forum_mentions"].max()), 1.0)
        forum_score["forum_score"] = (forum_score["forum_mentions"] / max_mentions * 100).round(1)
        grouped = grouped.merge(forum_score, on="symbol", how="left")
    else:
        grouped["forum_score"] = 0
        grouped["forum_mentions"] = 0
    grouped[["forum_score", "forum_mentions"]] = grouped[["forum_score", "forum_mentions"]].fillna(0)

    penalty_map = {"high": 30, "medium": 15, "low": 5}
    if not risk.empty and "symbol" in risk.columns:
        risk_penalty = risk.copy()
        risk_penalty["symbol"] = risk_penalty["symbol"].astype(str)
        risk_penalty["risk_penalty"] = risk_penalty.get("risk_level", "low").astype(str).map(penalty_map).fillna(5)
        risk_penalty = risk_penalty.groupby("symbol", as_index=False).agg(risk_penalty=("risk_penalty", "max"))
        grouped = grouped.merge(risk_penalty, on="symbol", how="left")
    else:
        grouped["risk_penalty"] = 0
    grouped["risk_penalty"] = grouped["risk_penalty"].fillna(0)

    grouped["total_score"] = (
        grouped["etf_score"] * 0.58
        + grouped["forum_score"] * 0.15
        + grouped["avg_weight"].clip(0, 8) * 2.2
        - grouped["risk_penalty"]
    ).round(1).clip(0, 100)

    def conclusion(row: pd.Series) -> str:
        if row["risk_penalty"] >= 25:
            return "風險優先，不追"
        if row["total_score"] >= 78:
            return "強勢觀察"
        if row["total_score"] >= 68:
            return "觀察池"
        if row["total_score"] >= 56:
            return "題材追蹤"
        return "僅追蹤"

    def why(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["etf_count"] >= 3:
            reasons.append(f"{int(row['etf_count'])} 檔主動 ETF 同步")
        if row["weight_3d_change"] > 0:
            reasons.append(f"3日權重 +{row['weight_3d_change']:.2f}")
        if row["weight_10d_change"] > 0:
            reasons.append(f"10日權重 +{row['weight_10d_change']:.2f}")
        if row["forum_mentions"] > 0:
            reasons.append(f"討論 {int(row['forum_mentions'])} 次")
        if row["risk_penalty"] > 0:
            reasons.append(f"風險扣分 {int(row['risk_penalty'])}")
        return "；".join(reasons[:4]) if reasons else "資料不足"

    grouped["conclusion"] = grouped.apply(conclusion, axis=1)
    grouped["why"] = grouped.apply(why, axis=1)
    return grouped.sort_values("total_score", ascending=False).reset_index(drop=True)


def sector_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["sector", "etf_strength", "attention", "risk_penalty", "total", "symbols"])
    base = (
        scores.groupby("sector", as_index=False)
        .agg(
            etf_strength=("etf_score", "mean"),
            attention=("forum_score", "mean"),
            risk_penalty=("risk_penalty", "mean"),
            stock_count=("symbol", "count"),
            symbols=("symbol", lambda x: "、".join(map(str, list(x)[:5]))),
        )
    )
    base["total"] = (base["etf_strength"] * 0.70 + base["attention"] * 0.18 - base["risk_penalty"] * 0.35).round(1).clip(0, 100)
    for col in ["etf_strength", "attention", "risk_penalty"]:
        base[col] = base[col].round(1)
    return base.sort_values("total", ascending=False).reset_index(drop=True)


def market_evidence(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = ["模組", "客觀結果", "方向", "分數", "證據"]
    if market.empty:
        return pd.DataFrame(columns=columns), {
            "verdict": "資料不足 / 不下結論",
            "class": "neutral",
            "one_line": "缺美股與台指夜盤資料，不能用 ETF 或 KOL 硬推方向。",
            "us_score": 0,
            "night_score": 0,
            "total_score": 0,
        }

    m = market.copy()
    m["item"] = m.get("item", "").astype(str)
    m["bias"] = m.get("bias", "neutral").astype(str).str.lower()
    m = normalize_numeric(m, ["value"])

    def score_from_bias(bias: str, value: float) -> int:
        if bias == "bullish":
            return 1
        if bias == "bearish":
            return -1
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    m["score"] = m.apply(lambda r: score_from_bias(str(r.get("bias", "neutral")), float(r.get("value", 0))), axis=1)
    m["direction_text"] = m["score"].map({1: "偏多", 0: "中性", -1: "偏空"}).fillna("中性")

    night_mask = m["item"].str.contains("台指|夜盤|期", regex=True, na=False)
    us_mask = ~night_mask
    us_score = int(m.loc[us_mask, "score"].sum())
    night_score = int(m.loc[night_mask, "score"].sum())
    total = us_score + night_score

    if us_score <= -2 and night_score <= -1:
        verdict = "偏空 / 觀望"
        css_class = "bear"
        one_line = "簡單結論：隔夜風向與台指夜盤同向偏空，今天不應輸出『上漲』結論；先防守，不追高。"
    elif us_score <= -2 or night_score <= -1:
        verdict = "中性偏空"
        css_class = "neutral"
        one_line = "簡單結論：至少一個關鍵盤前模組偏空，今日先觀望，等 9:10 開盤確認。"
    elif us_score >= 3 and night_score >= 1:
        verdict = "偏多但需確認"
        css_class = "bull"
        one_line = "簡單結論：盤前證據偏多，但仍要等日盤開盤後確認，不直接追價。"
    elif total > 0:
        verdict = "中性偏多"
        css_class = "neutral"
        one_line = "簡單結論：證據略偏多，但沒有強到可以直接下多方結論。"
    else:
        verdict = "中性觀望"
        css_class = "neutral"
        one_line = "簡單結論：盤前證據不足或互相抵銷，今日先等開盤確認。"

    rows = []
    if not m.loc[us_mask].empty:
        us_notes = []
        for _, r in m.loc[us_mask].iterrows():
            sign = "+" if float(r.get("value", 0)) > 0 else ""
            us_notes.append(f"{r['item']} {sign}{float(r.get('value', 0)):.2f}%：{r.get('note', '')}")
        rows.append({
            "模組": "1. 前一日美股 / 費半",
            "客觀結果": "偏多數" if us_score > 0 else "偏空數" if us_score < 0 else "多空相抵",
            "方向": "偏多" if us_score > 0 else "偏空" if us_score < 0 else "中性",
            "分數": us_score,
            "證據": "｜".join(us_notes[:8]),
        })
    if not m.loc[night_mask].empty:
        nt_notes = []
        for _, r in m.loc[night_mask].iterrows():
            val = float(r.get("value", 0))
            sign = "+" if val > 0 else ""
            unit = "點" if abs(val) > 20 else "%"
            nt_notes.append(f"{r['item']} {sign}{val:.2f}{unit}：{r.get('note', '')}")
        rows.append({
            "模組": "2. 台指期夜盤",
            "客觀結果": "夜盤偏多" if night_score > 0 else "夜盤偏空" if night_score < 0 else "夜盤中性",
            "方向": "偏多" if night_score > 0 else "偏空" if night_score < 0 else "中性",
            "分數": night_score,
            "證據": "｜".join(nt_notes[:4]),
        })

    return pd.DataFrame(rows, columns=columns), {
        "verdict": verdict,
        "class": css_class,
        "one_line": one_line,
        "us_score": us_score,
        "night_score": night_score,
        "total_score": total,
    }


def build_evidence_rows(scores: pd.DataFrame, sectors: pd.DataFrame, risk: pd.DataFrame, forum: pd.DataFrame, market_ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not market_ev.empty:
        rows.extend(market_ev.to_dict("records"))

    if sectors.empty:
        rows.append({"模組": "3. 主動 ETF 持股趨勢", "客觀結果": "資料不足", "方向": "不判斷", "分數": 0, "證據": "尚無 ETF 持股資料。"})
    else:
        s = sectors.iloc[0]
        rows.append({
            "模組": "3. 主動 ETF 持股趨勢",
            "客觀結果": f"熱區：{s['sector']}",
            "方向": "中期熱區，不等於今日方向",
            "分數": float(s["total"]),
            "證據": f"ETF 強度 {s['etf_strength']}；注意力 {s['attention']}；代表股 {s['symbols']}。",
        })

    risk_count = len(risk)
    high_risk = 0 if risk.empty or "risk_level" not in risk.columns else int((risk["risk_level"].astype(str) == "high").sum())
    rows.append({
        "模組": "4. 處置 / 注意股票",
        "客觀結果": f"{risk_count} 檔風險股，高風險 {high_risk} 檔",
        "方向": "風控扣分",
        "分數": -high_risk,
        "證據": "處置中、注意累計、即將解除只用於風險控管，不當作利多。",
    })

    if forum.empty:
        rows.append({"模組": "5. 論壇 / KOL 熱度", "客觀結果": "資料不足", "方向": "不判斷", "分數": 0, "證據": "尚無討論熱度資料。"})
    else:
        f = forum.copy()
        f["symbol"] = f.get("symbol", "").astype(str)
        f = normalize_numeric(f, ["mentions"])
        top = f.sort_values("mentions", ascending=False).head(3)
        rows.append({
            "模組": "5. 論壇 / KOL 熱度",
            "客觀結果": "注意力集中" if top["mentions"].sum() > 0 else "無明顯熱度",
            "方向": "注意力，不等於勝率",
            "分數": int(top["mentions"].sum()),
            "證據": "；".join((top["symbol"].astype(str) + " " + top.get("name", pd.Series([""] * len(top))).astype(str) + "：" + top["mentions"].astype(int).astype(str) + "次").tolist()),
        })
    return pd.DataFrame(rows)


def build_ai_conclusion(verdict: dict[str, object], sectors: pd.DataFrame, scores: pd.DataFrame, risk: pd.DataFrame) -> list[dict[str, str]]:
    top_sector = sectors.iloc[0]["sector"] if not sectors.empty else "尚未形成"
    top_symbols = sectors.iloc[0]["symbols"] if not sectors.empty else "尚無"
    watch = scores.head(5)
    watch_text = "、".join((watch["symbol"].astype(str) + " " + watch["name"].astype(str)).tolist()) if not watch.empty else "尚無"
    high_risk = 0 if risk.empty or "risk_level" not in risk.columns else int((risk["risk_level"].astype(str) == "high").sum())
    market_verdict = str(verdict["verdict"])

    if "偏空" in market_verdict:
        action = "不追高，觀察 9:10 是否收回夜盤壓力；若台積電、櫃買、電子權值沒有同步止穩，今日只做風險控管。"
    elif "偏多" in market_verdict:
        action = "可以準備觀察池，但只在開盤確認後處理；第一根急拉不追，等回測或突破有效。"
    else:
        action = "先等開盤確認，不用急著找多方標的。"

    return [
        {"title": "1. 簡單結論", "body": f"{market_verdict}。{action}"},
        {"title": "2. 客觀證據", "body": f"盤前方向由美股 / 費半與台指期夜盤決定；ETF、KOL、處置股不能覆蓋這個方向。美股分數 {verdict['us_score']}，夜盤分數 {verdict['night_score']}。"},
        {"title": "3. 資金熱區", "body": f"主動 ETF 與注意力目前指向「{top_sector}」，代表股：{top_symbols}。這是觀察池來源，不是今日一定上漲的理由。"},
        {"title": "4. 今日觀察池", "body": f"只列觀察，不列買進：{watch_text}。若大盤偏空，觀察池也要降級，只看誰抗跌、誰能在 9:10 後轉強。"},
        {"title": "5. 失效條件", "body": f"若台指期續弱、台積電開高走低、櫃買翻黑、主線族群只剩單點拉抬，偏多假設失效。處置 / 注意高風險 {high_risk} 檔不追。"},
    ]


def scenario_table(verdict: str) -> pd.DataFrame:
    if "偏空" in verdict:
        main_action = "防守：不追高，等 9:10 確認是否止穩。"
    elif "偏多" in verdict:
        main_action = "進攻但等確認：只看觀察池，等回測或突破有效。"
    else:
        main_action = "觀望：等開盤量價與權值股確認。"
    return pd.DataFrame([
        {"劇本": "目前主劇本", "確認條件": verdict, "動作": main_action},
        {"劇本": "轉強條件", "確認條件": "台指期收回夜盤關鍵價；台積電不開高走低；電子權值與櫃買同步轉強", "動作": "觀察池恢復有效，但仍不追第一根急拉"},
        {"劇本": "轉弱條件", "確認條件": "台指期跌破夜盤低點；台積電壓回；櫃買翻黑；強勢族群只剩單點", "動作": "取消追價，風險股與論壇過熱股列為不碰"},
    ])


bundle = load_sample_data()

with st.sidebar:
    st.title("資料設定")
    st.caption("第一版用 CSV / 範例資料。正式版會改成 8:30 自動抓資料、8:50 自動產報告、9:10 更新確認版。")
    mode = st.radio("報告模式", ["8:50 盤前版", "9:10 開盤確認版"], index=0)
    st.divider()
    with st.expander("進階：手動上傳 CSV", expanded=False):
        etf_df = upload_or_sample("ETF 持股 CSV", bundle.etf, "etf")
        risk_df = upload_or_sample("處置 / 注意股 CSV", bundle.risk, "risk")
        forum_df = upload_or_sample("論壇 / KOL CSV", bundle.forum, "forum")
        market_df = upload_or_sample("美股 / 台指期 CSV", bundle.market, "market")
    st.divider()
    st.caption("左側只放設定；主畫面先給結論，再給證據。")

scores = score_stocks(etf_df, risk_df, forum_df)
sectors = sector_scores(scores)
market_ev, verdict = market_evidence(market_df)
evidence = build_evidence_rows(scores, sectors, risk_df, forum_df, market_ev)
report_items = build_ai_conclusion(verdict, sectors, scores, risk_df)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">{APP_TITLE}</div>
      <div class="hero-sub">{mode}｜產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}｜流程：簡單結論 → 五模組客觀證據 → AI 綜合結論</div>
      <div style="margin-top:.75rem;">
        <span class="badge">1 美股 / 費半</span>
        <span class="badge">2 台指期夜盤</span>
        <span class="badge">3 主動 ETF</span>
        <span class="badge">4 處置注意股</span>
        <span class="badge">5 論壇 KOL</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

css_class = "bull" if verdict["class"] == "bull" else "neutral" if verdict["class"] == "neutral" else ""
st.markdown(f"<div class='verdict {css_class}'><b>{verdict['one_line']}</b></div>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("盤前結論", str(verdict["verdict"]))
m2.metric("美股分數", int(verdict["us_score"]))
m3.metric("夜盤分數", int(verdict["night_score"]))
m4.metric("ETF 熱區", sectors.iloc[0]["sector"] if not sectors.empty else "待確認")
m5.metric("風險股", len(risk_df))

st.subheader("先看五個模組的客觀結果")
st.caption("規則：美股 + 台指期夜盤決定今日盤前方向；ETF 只決定觀察池；KOL 只代表注意力；處置股只負責風險扣分。")
st.dataframe(evidence, use_container_width=True, hide_index=True)

st.subheader("AI 綜合後的五大結論")
for row1, row2 in zip(report_items[0::2], report_items[1::2]):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='card'><h4>{row1['title']}</h4><p>{row1['body']}</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h4>{row2['title']}</h4><p>{row2['body']}</p></div>", unsafe_allow_html=True)
if len(report_items) % 2 == 1:
    st.markdown(f"<div class='card'><h4>{report_items[-1]['title']}</h4><p>{report_items[-1]['body']}</p></div>", unsafe_allow_html=True)

st.subheader("今日交易劇本")
st.dataframe(scenario_table(str(verdict["verdict"])), use_container_width=True, hide_index=True)

st.divider()

tab_overview, tab_etf, tab_market, tab_risk, tab_forum, tab_model = st.tabs([
    "總覽", "主動 ETF 熱區", "美股 / 台指夜盤", "處置注意股", "論壇 / KOL", "規則模型"
])

with tab_overview:
    st.subheader("盤前總覽")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"<div class='evidence-card'><div class='tag'>第一優先：盤前方向</div><div class='big'>{verdict['verdict']}</div><div class='small'>美股與夜盤同向偏空時，不允許輸出偏多結論。</div></div>", unsafe_allow_html=True)
    with b:
        top = scores.iloc[0] if not scores.empty else None
        st.markdown(f"<div class='evidence-card'><div class='tag'>觀察池來源</div><div class='big'>{str(top['symbol']) + ' ' + str(top['name']) if top is not None else '待確認'}</div><div class='small'>{str(top['why']) if top is not None else '資料不足'}；大盤偏空時降級觀察。</div></div>", unsafe_allow_html=True)
    with c:
        st.markdown(f"<div class='evidence-card'><div class='tag'>風險控管</div><div class='big'>{len(risk_df)} 檔</div><div class='small'>處置 / 注意股不拿來當利多，只扣分或排除。</div></div>", unsafe_allow_html=True)
    st.write("")
    st.subheader("產業熱區矩陣")
    st.dataframe(sectors, use_container_width=True, hide_index=True)

with tab_etf:
    st.subheader("主動 ETF 共識觀察股")
    st.caption("ETF 熱區是中期資金線索，不負責判斷今天大盤會漲。")
    display_cols = ["symbol", "name", "sector", "etf_count", "avg_weight", "weight_3d_change", "weight_5d_change", "weight_10d_change", "forum_mentions", "risk_penalty", "total_score", "conclusion", "why"]
    st.dataframe(scores[display_cols] if not scores.empty else scores, use_container_width=True, hide_index=True)
    if not sectors.empty:
        st.bar_chart(sectors.set_index("sector")["total"])

with tab_market:
    st.subheader("隔夜美股與台指期夜盤")
    st.caption("這裡是盤前方向的第一優先證據。")
    st.dataframe(market_df, use_container_width=True, hide_index=True)

with tab_risk:
    st.subheader("處置 / 注意風險股票")
    if risk_df.empty:
        st.success("目前沒有處置 / 注意股資料。")
    else:
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

with tab_forum:
    st.subheader("論壇 / KOL 注意力")
    st.caption("注意力高不等於勝率高；高熱度 + 高漲幅反而可能是追高風險。")
    if forum_df.empty:
        st.info("尚無論壇資料。")
    else:
        f = forum_df.copy()
        f["symbol"] = f.get("symbol", "").astype(str)
        f = normalize_numeric(f, ["mentions"])
        if "topic" in f.columns:
            topic = f.groupby("topic", as_index=False).agg(mentions=("mentions", "sum")).sort_values("mentions", ascending=False)
            st.bar_chart(topic.set_index("topic")["mentions"])
        st.dataframe(f, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("規則模型")
    st.markdown(
        """
        **硬規則：**
        1. 如果美股 / 費半偏空，且台指期夜盤也偏空，盤前結論最高只能是「偏空 / 觀望」。
        2. ETF 熱區只能決定「觀察池」，不能把大盤方向從偏空改成偏多。
        3. KOL / 論壇只代表注意力，不能直接加成為買進訊號。
        4. 處置 / 注意股只做風險扣分，不當成利多。
        5. 9:10 版才允許根據台積電、電子權值、櫃買、台指期日盤重新調整劇本。

        **下一階段：** 接正式資料源後，每天記錄「8:50 結論 → 9:10 確認 → 收盤結果」，才能回測分數是否有效。
        """
    )

st.divider()
st.markdown("<span class='small-muted'>MVP：研究儀表板，不是投資建議。正式公開前需補資料授權、回測、免責聲明與排程。</span>", unsafe_allow_html=True)
