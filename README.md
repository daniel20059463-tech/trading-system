# 台股每日預測與驗證系統

這個專案整合台股歷史行情、盤前新聞、前一晚美股訊號、每日重訓候選、盤中監控及盤後準確度驗證。

## 在另一台 Windows 電腦使用

1. 安裝 Git、Python 3.11 與必要的 Microsoft C++ 執行環境。
2. 複製私人儲存庫：

   ```powershell
   git clone <這個私人儲存庫網址>
   cd trading-system
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. 將 `.env.example` 複製成 `.env`，再自行填入 API 金鑰、FinMind token 與 Discord webhook。`.env` 不會上傳 GitHub。
4. 先手動執行健康檢查與測試，再建立該電腦自己的 Windows 排程。

   ```powershell
   .\health_check.ps1
   python -m pytest -q
   ```

## 兩台電腦同步

開始工作前：

```powershell
git pull --rebase
```

完成修改或產生需要共同保存的新資料後：

```powershell
git add -A
git commit -m "更新模型與實證資料"
git push
```

為避免重複推播與兩套模型同時寫入不同結果，08:30、08:36、12:00、15:30 的正式排程應只在一台主要電腦啟用。另一台電腦用於開發、查閱與備援；切換主要電腦前，先在舊電腦停用排程、推送最新資料，再由新電腦拉取。

## 不會同步的內容

- `.env` 與所有金鑰
- 執行紀錄 `logs/`
- Python 快取與本機安裝的套件
- Windows 排程備份

歷史行情、新聞資料、預測、模型、實證與報告會納入版本控制，供另一台電腦接續使用。
