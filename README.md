# 台股 8:50 盤前戰情室 — Streamlit 部署版

這是可以直接部署到 Streamlit Community Cloud 的 MVP 版本。

## 檔案

- `app.py`：Streamlit 主程式
- `requirements.txt`：部署依賴
- `data/*.csv`：範例資料
- `.streamlit/config.toml`：深色主題設定

## Streamlit Cloud 設定

- Repository：你的 GitHub repo
- Branch：main
- Main file path：app.py
- App URL：可填 `tw-stock-war-room` 或你想要的名稱

## 注意

這是研究儀表板，不是投資建議。第一版先用範例資料與 CSV 上傳機制，後續再接正式資料源。
