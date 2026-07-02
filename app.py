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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --bg: #0d0a05;
  --card: #17130d;
  --gold: #d4af37;
  --cream: #f5e6c8;
  --muted: #b8aa8a;
  --danger: #c96a5b;
}
.stApp {
  background: linear-gradient(135deg, #0d0a05 0%, #17130d 52%, #241f16 100%);
  color: var(--cream);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--cream); }
[data-testid="stMetricValue"] { color: var(--cream); }
[data-testid="stMetricLabel"] { color: var(--muted); }
.card {
  border: 1px solid rgba(212, 175, 55, .25);
  background: rgba(23, 19, 13, .82);
  border-radius: 18px;
  padding: 1.05rem 1.15rem;
  box-shadow: 0 12px 35px rgba(0,0,0,.22);
  min-height: 120px;
}
.card h4 { color: var(--gold); margin: 0 0 .5rem 0; font-size: 1.02rem; }
.card p { color: var(--cream); margin: .25rem 0; line-height: 1.55; }
.badge {
  display:inline-block; padding:.25rem .55rem; border-radius:999px;
  border:1px solid rgba(212,175,55,.35); color:var(--gold); font-size:.82rem;
}
.small-muted { color: var(--muted); font-size:.88rem; }
.warning-box {
  border-left: 4px solid var(--danger);
  background: rgba(201,106,91,.12);
  padding:.85rem 1rem;
  border-radius:12px;
  color:var(--cream);
}
hr { border-color: rgba(212,175,55,.15); }
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
    return pd.read_csv(path)


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


def score_stocks(etf: pd.DataFrame, risk: pd.DataFrame, forum: pd.DataFrame) -> pd.DataFrame:
    if etf.empty:
        return pd.DataFrame(columns=[
            "symbol", "name", "sector", "etf_count", "avg_weight", "trend_score",
            "forum_score", "risk_penalty", "total_score", "conclusion"
        ])

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

    grouped["trend_score"] = (
        grouped["etf_count"].clip(0, 4) * 12
        + grouped["weight_3d_change"].clip(-1, 1) * 25
        + grouped["weight_5d_change"].clip(-1, 1) * 22
        + grouped["weight_10d_change"].clip(-1, 1) * 18
        + (grouped["shares_change_10d"] > 0).astype(int) * 8
    ).clip(0, 100)

    if not forum.empty and "symbol" in forum.columns:
        forum = normalize_numeric(forum, ["mentions"])
        forum_score = (
            forum.groupby("symbol", as_index=False)
            .agg(forum_mentions=("mentions", "sum"))
        )
        max_mentions = max(float(forum_score["forum_mentions"].max()), 1.0)
        forum_score["forum_score"] = (forum_score["forum_mentions"] / max_mentions * 100).round(1)
        grouped = grouped.merge(forum_score[["symbol", "forum_score", "forum_mentions"]], on="symbol", how="left")
    else:
        grouped["forum_score"] = 0
        grouped["forum_mentions"] = 0

    grouped["forum_score"] = grouped["forum_score"].fillna(0)
    grouped["forum_mentions"] = grouped["forum_mentions"].fillna(0)

    penalty_map = {"high": 30, "medium": 15, "low": 5}
    if not risk.empty and "symbol" in risk.columns:
        risk_penalty = risk.copy()
        risk_penalty["risk_penalty"] = risk_penalty.get("risk_level", "low").map(penalty_map).fillna(5)
        risk_penalty = risk_penalty.groupby("symbol", as_index=False).agg(risk_penalty=("risk_penalty", "max"))
        grouped = grouped.merge(risk_penalty, on="symbol", how="left")
    else:
        grouped["risk_penalty"] = 0
    grouped["risk_penalty"] = grouped["risk_penalty"].fillna(0)

    grouped["total_score"] = (
        grouped["trend_score"] * 0.72
        + grouped["forum_score"] * 0.18
        + grouped["avg_weight"].clip(0, 5) * 2
        - grouped["risk_penalty"]
    ).round(1).clip(0, 100)

    def conclusion(row: pd.Series) -> str:
        if row["risk_penalty"] >= 25:
            return "風險優先，不追"
        if row["total_score"] >= 80:
            return "強勢觀察"
        if row["total_score"] >= 70:
            return "今日觀察池"
        if row["total_score"] >= 60:
            return "有題材但確認不足"
        return "僅追蹤"

    grouped["conclusion"] = grouped.apply(conclusion, axis=1)
    return grouped.sort_values("total_score", ascending=False).reset_index(drop=True)


def sector_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["sector", "avg_score", "stock_count", "symbols"])
    out = (
        scores.groupby("sector", as_index=False)
        .agg(
            avg_score=("total_score", "mean"),
            stock_count=("symbol", "count"),
            symbols=("symbol", lambda x: "、".join(map(str, list(x)[:5]))),
        )
        .sort_values("avg_score", ascending=False)
    )
    out["avg_score"] = out["avg_score"].round(1)
    return out


def market_bias(market: pd.DataFrame, scores: pd.DataFrame, risk: pd.DataFrame) -> str:
    bias_score = 0
    if not market.empty and "bias" in market.columns:
        for b in market["bias"].dropna().astype(str):
            if b == "bullish":
                bias_score += 1
            elif b == "bearish":
                bias_score -= 1
    if not scores.empty and (scores["total_score"] >= 70).sum() >= 3:
        bias_score += 1
    if not risk.empty and "risk_level" in risk.columns and (risk["risk_level"] == "high").sum() >= 8:
        bias_score -= 1
    if bias_score >= 4:
        return "偏多"
    if bias_score <= -2:
        return "偏空"
    return "中性偏觀察"


def build_conclusions(mode: str, bias: str, sectors: pd.DataFrame, scores: pd.DataFrame, risk: pd.DataFrame, market: pd.DataFrame) -> list[str]:
    top_sector = sectors.iloc[0]["sector"] if not sectors.empty else "尚未形成明確主線"
    top_symbols = "、".join(scores.head(5)["symbol"].astype(str).tolist()) if not scores.empty else "尚無"
    risk_count = len(risk)
    tx_note = ""
    if not market.empty and "item" in market.columns:
        tx_rows = market[market["item"].astype(str).str.contains("台指", na=False)]
        if not tx_rows.empty:
            tx_note = str(tx_rows.iloc[0].get("note", ""))
    mode_note = "8:50 盤前版" if mode == "premarket" else "9:10 開盤確認版"
    return [
        f"{mode_note}市場方向為「{bias}」；這不是追價訊號，仍要看日盤開盤後權值股是否延續。",
        f"目前資金熱區以「{top_sector}」最明顯；若該族群開高走低，今日主線可信度要下修。",
        f"主動 ETF / 論壇熱度綜合觀察股：{top_symbols}；這是觀察池，不是買進清單。",
        f"處置 / 注意風險股票目前偵測 {risk_count} 檔；處置中與注意累計股不追高。",
        f"失效條件：台指期跌破夜盤關鍵價、台積電開高走低、櫃買翻黑，則今日強勢劇本降級。{tx_note}",
    ]


def upload_or_sample(label: str, sample: pd.DataFrame, key: str) -> pd.DataFrame:
    file = st.sidebar.file_uploader(label, type=["csv"], key=key)
    if file is not None:
        try:
            return pd.read_csv(file)
        except Exception as exc:
            st.sidebar.error(f"CSV 讀取失敗：{exc}")
    return sample


bundle = load_sample_data()

with st.sidebar:
    st.title("資料設定")
    st.caption("第一版可先用範例資料部署，之後再接正式資料源。")
    mode = st.radio("報告模式", ["premarket", "open-confirm"], format_func=lambda x: "8:50 盤前版" if x == "premarket" else "9:10 開盤確認版")
    st.divider()
    etf_df = upload_or_sample("上傳 ETF 持股 CSV", bundle.etf, "etf")
    risk_df = upload_or_sample("上傳 處置/注意股 CSV", bundle.risk, "risk")
    forum_df = upload_or_sample("上傳 論壇/KOL CSV", bundle.forum, "forum")
    market_df = upload_or_sample("上傳 美股/台指期 CSV", bundle.market, "market")
    st.divider()
    st.caption("正式版會改成排程自動抓資料；目前先把網站部署成功。")

scores = score_stocks(etf_df, risk_df, forum_df)
sectors = sector_scores(scores)
bias = market_bias(market_df, scores, risk_df)
conclusions = build_conclusions(mode, bias, sectors, scores, risk_df, market_df)

st.title(APP_TITLE)
st.caption(f"自用研究儀表板｜產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}｜資料狀態：範例資料 / 可手動上傳 CSV")

st.markdown("<div class='warning-box'>這是研究儀表板，不是投資建議。主動 ETF 權重變化不等於精準買賣超；KOL / 論壇訊號只代表注意力，不代表勝率。</div>", unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("今日市場方向", bias)
c2.metric("主線產業", sectors.iloc[0]["sector"] if not sectors.empty else "待確認")
c3.metric("強勢觀察股", int((scores["total_score"] >= 70).sum()) if not scores.empty else 0)
c4.metric("風險股票", len(risk_df))

st.subheader("五大結論")
for i, item in enumerate(conclusions, start=1):
    st.markdown(f"<div class='card'><h4>{i}. 結論</h4><p>{item}</p></div>", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["資金熱區", "觀察股排名", "風險股", "美股/夜盤", "論壇/KOL"])

with tab1:
    st.subheader("產業熱區分數")
    if sectors.empty:
        st.info("尚無產業資料。")
    else:
        st.bar_chart(sectors.set_index("sector")["avg_score"])
        st.dataframe(sectors, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("主動 ETF 共識觀察股")
    cols = ["symbol", "name", "sector", "etf_count", "avg_weight", "trend_score", "forum_score", "risk_penalty", "total_score", "conclusion"]
    st.dataframe(scores[cols] if not scores.empty else scores, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("處置 / 注意風險股票")
    if risk_df.empty:
        st.success("目前沒有風險股資料。")
    else:
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("隔夜美股與台指期夜盤")
    if market_df.empty:
        st.info("尚無市場資料。")
    else:
        st.dataframe(market_df, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("論壇 / KOL 注意力")
    if forum_df.empty:
        st.info("尚無論壇資料。")
    else:
        top_forum = normalize_numeric(forum_df, ["mentions"])
        if "topic" in top_forum.columns:
            topic = top_forum.groupby("topic", as_index=False).agg(mentions=("mentions", "sum")).sort_values("mentions", ascending=False)
            st.bar_chart(topic.set_index("topic")["mentions"])
        st.dataframe(top_forum, use_container_width=True, hide_index=True)

st.divider()
st.markdown("<span class='small-muted'>MVP 下一階段：接 TWSE/TPEx 處置注意股、TAIFEX 台指期夜盤、主動 ETF 每日持股來源、論壇標題爬取與 8:50 自動排程。</span>", unsafe_allow_html=True)
