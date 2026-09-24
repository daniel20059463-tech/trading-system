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

## 每日候選採用治理

每日重訓產生的是影子候選。盤後系統只根據盤前不可覆寫的同股、同日預測及實際收盤計分；候選若同時滿足至少20個交易日／200筆、方向與誤差門檻、區間涵蓋，以及至少60%的逐日勝出率，才進入 `eligible_for_review`。移除任一天後平均優勢仍須為正，以避免單日行情主導結果。

```powershell
python adoption_state.py status
python adoption_state.py review
```

待審狀態會列出完整事件雜湊、計分日、受影響股票、候選模型、原正式模型與規則版本。檢視後若決定採用，將待審事件中的 `event_sha256` 貼入：

```powershell
python adoption_state.py promote --expected-event-hash <待審事件的完整雜湊>
```

若資料或版本在檢視後變動，命令會拒絕使用過期事件。採用從下一個交易日開始；十檔候選、原正式模型與雜湊必須同時完整，才會替換正式預測並重算策略。盤前預測與策略會帶有採用 ID、模型與規則版本，盤後成績另存為不可覆寫結算。至少10個交易日／100筆後，若誤差高於原模型或猜不變基準2%且至少60%的日子未勝出，或方向準確率低於原模型5個百分點，就自動回滾。保留決策至少需20個交易日／200筆的獨立採用後樣本。也可以手動回滾：

```powershell
python adoption_state.py rollback --reason "人工檢查發現退化"
```

事件保存在 `live_evidence/adoption/events/`，含前一事件雜湊；監測結算保存在 `live_evidence/adoption/settlements/`。回滾後停止這一輪採用；新候選使用不同協定版本並備好模型後，執行 `python adoption_state.py restart`，才重新進入 shadow 驗證。
