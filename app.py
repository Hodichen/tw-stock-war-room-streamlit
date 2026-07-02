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
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root {
  --bg: #0d0a05;
  --bg2: #17130d;
  --card: rgba(24, 19, 12, .92);
  --card2: rgba(38, 29, 18, .72);
  --gold: #d4af37;
  --gold2: #f3d37a;
  --cream: #f5e6c8;
  --muted: #b8aa8a;
  --line: rgba(212,175,55,.22);
  --danger: #c96a5b;
  --green: #81c995;
}
.stApp {
  background: radial-gradient(circle at 20% 0%, #2a2113 0%, #0d0a05 34%, #090704 100%);
  color: var(--cream);
}
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #f3f4f7; }
section[data-testid="stSidebar"] * { color: #1f2937 !important; }
h1, h2, h3 { color: var(--cream); letter-spacing: .02em; }
[data-testid="stMetric"] {
  border: 1px solid var(--line);
  background: rgba(23,19,13,.76);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  min-height: 122px;
}
[data-testid="stMetricValue"] { color: var(--cream); font-size: 2.1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
.hero {
  border: 1px solid rgba(212,175,55,.32);
  background: linear-gradient(135deg, rgba(212,175,55,.18), rgba(23,19,13,.72));
  border-radius: 26px;
  padding: 1.45rem 1.65rem;
  box-shadow: 0 18px 50px rgba(0,0,0,.26);
  margin-bottom: 1.05rem;
}
.hero-title { font-size: 2.25rem; font-weight: 800; color: var(--cream); margin-bottom: .35rem; }
.hero-sub { color: var(--muted); font-size: .98rem; line-height: 1.65; }
.notice {
  border-left: 5px solid var(--danger);
  background: rgba(201,106,91,.13);
  padding: .8rem 1rem;
  border-radius: 14px;
  color: var(--cream);
  margin: .75rem 0 1.1rem 0;
}
.card {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 20px;
  padding: 1.05rem 1.15rem;
  box-shadow: 0 12px 35px rgba(0,0,0,.24);
  min-height: 136px;
  margin-bottom: .75rem;
}
.card h4 { color: var(--gold2); margin: 0 0 .55rem 0; font-size: 1.05rem; }
.card p { color: var(--cream); margin: .25rem 0; line-height: 1.7; }
.signal-card {
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(38,29,18,.88), rgba(16,12,8,.88));
  border-radius: 20px;
  padding: 1rem 1.05rem;
  min-height: 150px;
}
.signal-card .tag { color: var(--gold); font-size:.88rem; font-weight: 700; }
.signal-card .big { color: var(--cream); font-size: 1.35rem; font-weight: 800; margin:.45rem 0; }
.signal-card .small { color: var(--muted); line-height:1.55; font-size:.92rem; }
.badge {
  display:inline-block; padding:.23rem .58rem; border-radius:999px;
  border:1px solid rgba(212,175,55,.38); color:var(--gold2); font-size:.82rem;
  background: rgba(212,175,55,.07); margin-right:.35rem;
}
.small-muted { color: var(--muted); font-size:.88rem; }
hr { border-color: rgba(212,175,55,.12); }
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
            return "今日觀察池"
        if row["total_score"] >= 56:
            return "有題材，待開盤確認"
        return "僅追蹤"

    def why(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["etf_count"] >= 3:
            reasons.append(f"{int(row['etf_count'])} 檔主動 ETF 同步持有/加權")
        if row["weight_3d_change"] > 0:
            reasons.append(f"3日權重 +{row['weight_3d_change']:.2f}")
        if row["weight_10d_change"] > 0:
            reasons.append(f"10日權重 +{row['weight_10d_change']:.2f}")
        if row["forum_mentions"] > 0:
            reasons.append(f"論壇/KOL {int(row['forum_mentions'])} 次提及")
        if row["risk_penalty"] > 0:
            reasons.append(f"風險扣分 {int(row['risk_penalty'])}")
        return "；".join(reasons[:4]) if reasons else "資料不足，僅保留追蹤"

    grouped["conclusion"] = grouped.apply(conclusion, axis=1)
    grouped["why"] = grouped.apply(why, axis=1)
    return grouped.sort_values("total_score", ascending=False).reset_index(drop=True)


def sector_scores(scores: pd.DataFrame, forum: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
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
    base["total"] = (base["etf_strength"] * 0.68 + base["attention"] * 0.18 - base["risk_penalty"] * 0.35).round(1).clip(0, 100)
    for col in ["etf_strength", "attention", "risk_penalty"]:
        base[col] = base[col].round(1)
    return base.sort_values("total", ascending=False).reset_index(drop=True)


def market_interpretation(market: pd.DataFrame) -> tuple[str, int, int, pd.DataFrame]:
    if market.empty:
        return "待確認", 0, 0, pd.DataFrame()
    m = market.copy()
    m["item"] = m.get("item", "").astype(str)
    m["bias"] = m.get("bias", "neutral").astype(str)
    bullish = int((m["bias"] == "bullish").sum())
    bearish = int((m["bias"] == "bearish").sum())
    score = bullish - bearish
    if score >= 4:
        label = "偏多"
    elif score <= -2:
        label = "偏空"
    else:
        label = "中性偏觀察"
    return label, bullish, bearish, m


def build_report(mode_label: str, bias: str, sectors: pd.DataFrame, scores: pd.DataFrame, risk: pd.DataFrame, market: pd.DataFrame) -> list[dict[str, str]]:
    top_sector = sectors.iloc[0]["sector"] if not sectors.empty else "尚未形成明確主線"
    top_sector_symbols = sectors.iloc[0]["symbols"] if not sectors.empty else "尚無"
    top_watch = scores.head(5) if not scores.empty else pd.DataFrame()
    watch_text = "、".join((top_watch["symbol"].astype(str) + " " + top_watch["name"].astype(str)).tolist()) if not top_watch.empty else "尚無"
    high_risk = 0 if risk.empty or "risk_level" not in risk.columns else int((risk["risk_level"].astype(str) == "high").sum())
    tx_note = ""
    if not market.empty and "item" in market.columns:
        tx = market[market["item"].astype(str).str.contains("台指", na=False)]
        if not tx.empty:
            tx_note = str(tx.iloc[0].get("note", ""))

    prefix = "8:50 盤前版" if mode_label == "8:50 盤前版" else "9:10 開盤確認版"
    return [
        {
            "title": "1. 今日市場方向",
            "body": f"{prefix}暫判為「{bias}」。這不是進場訊號，而是盤前劇本；要等日盤開盤後用台積電、電子權值、櫃買與台指期價差確認。",
        },
        {
            "title": "2. 今日資金熱區",
            "body": f"目前主線產業以「{top_sector}」最明顯，代表 ETF 權重變化與市場注意力較集中。代表觀察股：{top_sector_symbols}。若族群開高走低，主線可信度要下修。",
        },
        {
            "title": "3. 主動 ETF 共識觀察池",
            "body": f"今日先觀察：{watch_text}。這裡不是買進清單，篩選邏輯是多檔主動 ETF 共識、3/5/10 日權重變化與論壇注意力綜合。",
        },
        {
            "title": "4. 風險與不追清單",
            "body": f"目前偵測 {len(risk)} 檔處置/注意相關股票，其中高風險 {high_risk} 檔。處置中、注意累計、論壇過熱但 ETF 沒確認者，不追高。",
        },
        {
            "title": "5. 今日失效條件",
            "body": f"若台指期跌破夜盤關鍵價、台積電開高走低、櫃買翻黑，或強勢族群只剩單點表態，今日偏多劇本降級。{tx_note}",
        },
    ]


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"劇本": "偏多劇本", "確認條件": "台指期站穩夜盤收盤價；台積電/電子權值不開高走低；主線族群有 2 檔以上續強", "動作": "只看觀察池，等回測或突破確認，不追第一根急拉"},
        {"劇本": "中性劇本", "確認條件": "美股偏多但台股開盤無量；櫃買或題材股不同步", "動作": "縮小部位，觀察到 9:10 後再判斷"},
        {"劇本": "偏空劇本", "確認條件": "台指期跌破夜盤低點；台積電壓回；櫃買翻黑", "動作": "取消追價，風險股與論壇過熱股列入不碰清單"},
    ])


bundle = load_sample_data()

with st.sidebar:
    st.title("資料設定")
    st.caption("預設使用範例資料；正式版會改成 8:30 自動抓資料、8:50 自動產報告、9:10 更新確認版。")
    mode = st.radio("報告模式", ["8:50 盤前版", "9:10 開盤確認版"], index=0)
    st.divider()
    with st.expander("進階：手動上傳 CSV", expanded=False):
        etf_df = upload_or_sample("ETF 持股 CSV", bundle.etf, "etf")
        risk_df = upload_or_sample("處置 / 注意股 CSV", bundle.risk, "risk")
        forum_df = upload_or_sample("論壇 / KOL CSV", bundle.forum, "forum")
        market_df = upload_or_sample("美股 / 台指期 CSV", bundle.market, "market")
    st.divider()
    st.caption("左側已改成收合式，不再把上傳框全部攤開。")

# If sidebar expander was never opened, variables are still created because Streamlit executes its body.
scores = score_stocks(etf_df, risk_df, forum_df)
sectors = sector_scores(scores, forum_df, risk_df)
bias, bullish_count, bearish_count, market_view = market_interpretation(market_df)
report_items = build_report(mode, bias, sectors, scores, risk_df, market_df)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">{APP_TITLE}</div>
      <div class="hero-sub">{mode}｜產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}｜定位：自用研究儀表板，不是投資建議</div>
      <div style="margin-top:.75rem;">
        <span class="badge">主動 ETF 共識</span>
        <span class="badge">美股 / 費半映射</span>
        <span class="badge">台指期夜盤</span>
        <span class="badge">處置 / 注意風險</span>
        <span class="badge">論壇 / KOL 注意力</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='notice'>這版已改成『盤前作戰儀表板』：先看市場劇本、資金熱區、共識觀察池、風險清單與失效條件。左側上傳功能已收合，不再干擾主畫面。</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("市場方向", bias)
m2.metric("主線產業", sectors.iloc[0]["sector"] if not sectors.empty else "待確認")
m3.metric("觀察股", int((scores["total_score"] >= 68).sum()) if not scores.empty else 0)
m4.metric("風險股", len(risk_df))
m5.metric("美股/夜盤偏多數", bullish_count)

st.subheader("五大結論")
for row1, row2 in zip(report_items[0::2], report_items[1::2]):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='card'><h4>{row1['title']}</h4><p>{row1['body']}</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h4>{row2['title']}</h4><p>{row2['body']}</p></div>", unsafe_allow_html=True)
if len(report_items) % 2 == 1:
    st.markdown(f"<div class='card'><h4>{report_items[-1]['title']}</h4><p>{report_items[-1]['body']}</p></div>", unsafe_allow_html=True)

st.subheader("今日交易劇本")
st.dataframe(scenario_table(), use_container_width=True, hide_index=True)

st.divider()

tab_overview, tab_etf, tab_market, tab_risk, tab_forum, tab_model = st.tabs([
    "總覽", "主動 ETF 熱區", "美股 / 台指夜盤", "處置注意股", "論壇 / KOL", "訊號模型"
])

with tab_overview:
    st.subheader("盤前總覽")
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"<div class='signal-card'><div class='tag'>資金熱區</div><div class='big'>{sectors.iloc[0]['sector'] if not sectors.empty else '待確認'}</div><div class='small'>看 3/5/10 日 ETF 權重變化，不只看單日。</div></div>", unsafe_allow_html=True)
    with b:
        top = scores.iloc[0] if not scores.empty else None
        st.markdown(f"<div class='signal-card'><div class='tag'>最高分觀察股</div><div class='big'>{str(top['symbol']) + ' ' + str(top['name']) if top is not None else '待確認'}</div><div class='small'>{str(top['why']) if top is not None else '資料不足'}</div></div>", unsafe_allow_html=True)
    with c:
        st.markdown(f"<div class='signal-card'><div class='tag'>風險提醒</div><div class='big'>{len(risk_df)} 檔</div><div class='small'>處置中、今日解除、即將解除、注意股只做風險控管，不當作利多。</div></div>", unsafe_allow_html=True)
    st.write("")
    st.subheader("產業熱區矩陣")
    st.dataframe(sectors, use_container_width=True, hide_index=True)

with tab_etf:
    st.subheader("主動 ETF 共識觀察股")
    st.caption("重點不是單一 ETF 買什麼，而是多檔主動 ETF 是否形成共識，且權重變化是否延續。")
    display_cols = ["symbol", "name", "sector", "etf_count", "avg_weight", "weight_3d_change", "weight_5d_change", "weight_10d_change", "forum_mentions", "risk_penalty", "total_score", "conclusion", "why"]
    st.dataframe(scores[display_cols] if not scores.empty else scores, use_container_width=True, hide_index=True)
    if not sectors.empty:
        st.bar_chart(sectors.set_index("sector")["total"])

with tab_market:
    st.subheader("隔夜美股與台指期夜盤")
    st.caption("這裡只判斷隔夜風向與台股映射，不直接產生買賣訊號。")
    if market_view.empty:
        st.info("尚無市場資料。")
    else:
        st.dataframe(market_view, use_container_width=True, hide_index=True)

with tab_risk:
    st.subheader("處置 / 注意風險股票")
    st.caption("處置解除不等於必漲；注意股也不是必跌。這區的用途是避免追到風險過高的標的。")
    if risk_df.empty:
        st.success("目前沒有處置 / 注意股資料。")
    else:
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

with tab_forum:
    st.subheader("論壇 / KOL 注意力")
    st.caption("論壇熱度只代表市場注意力，不代表勝率。高熱度 + 高漲幅反而要小心追高。")
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
    st.subheader("訊號模型與下一階段")
    st.markdown(
        """
        **目前模型：**
        - 主動 ETF 共識與 3/5/10 日權重變化：核心權重
        - 美股 / 費半 / 台指期夜盤：盤前方向確認
        - 處置 / 注意股：風險扣分，不拿來當利多
        - 論壇 / KOL：注意力分數，不等於買進訊號

        **下一階段要接的真實資料：**
        1. 主動 ETF 每日持股 / PCF
        2. TWSE / TPEx 注意股與處置股
        3. TAIFEX 台指期夜盤 OHLC
        4. 美股四大指數、費半、TSMC ADR、NVDA、AMD、AVGO
        5. PTT / Dcard / YouTube 標題熱度
        """
    )

st.divider()
st.markdown("<span class='small-muted'>MVP 版本：先驗證儀表板邏輯與使用流程。正式公開前，需要補資料授權、回測、免責聲明與排程。</span>", unsafe_allow_html=True)
