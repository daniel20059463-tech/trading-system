"""實驗：把每日新聞分數(news_score)當第20特徵，量測隔日漲跌方向準確率是否提升。
不動生產程式。公平比較：同一隨機種子下，唯一差別就是多/少 news_score 這一欄。
多種子取平均，避免單次雜訊誤判。"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import config as cfg
from models.lstm_model import get_model
from models.train import FEATURE_COLS

DEVICE = torch.device("cpu")
W = cfg.WINDOW_SIZE
SEEDS = [0, 1, 2]
EPOCHS = 80

nf = pd.read_csv("data/news_feature.csv")


def _prep(ticker, use_news):
    df = pd.read_csv(f"data/raw/{ticker}.csv", index_col=0, parse_dates=True)
    code = ticker.replace(".TWO", "").replace(".TW", "")
    ns = nf[nf["code"].astype(str) == code].set_index("date")["news_score"]
    df["news_score"] = df.index.map(lambda d: float(ns.get(str(d.date()), 0.0)))

    feats = [c for c in FEATURE_COLS if c in df.columns]
    cols = feats + ["news_score"]                 # 一律保留 news_score（供遮罩）
    sub = df[cols].dropna()
    news_col = sub["news_score"].values
    feat_cols = feats + (["news_score"] if use_news else [])
    data = sub[feat_cols].values.astype(np.float32)
    tgt = feat_cols.index("pct_change")

    split = int(len(data) * cfg.TRAIN_RATIO)
    scaler = MinMaxScaler().fit(data[:split])
    ds = scaler.transform(data)

    X, yreal, news_last = [], [], []
    for i in range(len(ds) - W):
        X.append(ds[i:i+W]); yreal.append(data[i+W, tgt])     # 真實 pct_change（未縮放）
        news_last.append(news_col[i + W - 1])                 # 窗口最後一天的新聞分數
    X = np.array(X, np.float32); yreal = np.array(yreal, np.float32)
    news_last = np.array(news_last, np.float32)
    sp = int(len(X) * cfg.TRAIN_RATIO)
    ymin, ymax = yreal[:sp].min(), yreal[:sp].max()
    ys = ((yreal - ymin) / (ymax - ymin + 1e-9)).reshape(-1, 1)
    return X[:sp], ys[:sp], X[sp:], yreal[sp:], (ymin, ymax), news_last[sp:]


def _train_eval(ticker, use_news, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte_real, (ymin, ymax), news_last = _prep(ticker, use_news)
    model = get_model("lstm", Xtr.shape[2], cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=cfg.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).cpu().numpy().ravel()
    pred_real = pred * (ymax - ymin) + ymin
    hit = np.sign(pred_real) == np.sign(yte_real)
    acc = hit.mean() * 100
    base = max((yte_real > 0).mean(), (yte_real < 0).mean()) * 100
    mask = news_last != 0                                  # 有新聞的測試日
    acc_news = hit[mask].mean() * 100 if mask.sum() > 0 else np.nan
    return acc, base, acc_news, int(mask.sum())


print(f"新聞特徵實驗（{len(SEEDS)} 種子平均，隔日方向命中率）")
print(f"{'股票':<10}{'無新聞':>8}{'有新聞':>8}{'差異':>8}│{'有新聞天:無':>11}{'有新聞天:有':>11}{'n':>5}")
print("-" * 62)
sum_no, sum_yes = [], []
sum_nd_no, sum_nd_yes = [], []
for tk in cfg.ALL_STOCKS:
    try:
        res_no = [_train_eval(tk, False, s) for s in SEEDS]
        res_yes = [_train_eval(tk, True, s) for s in SEEDS]
        a_no = np.mean([r[0] for r in res_no]); a_yes = np.mean([r[0] for r in res_yes])
        nd_no = np.nanmean([r[2] for r in res_no]); nd_yes = np.nanmean([r[2] for r in res_yes])
        n = res_yes[0][3]
        sum_no.append(a_no); sum_yes.append(a_yes)
        sum_nd_no.append(nd_no); sum_nd_yes.append(nd_yes)
        diff = a_yes - a_no
        mark = " ✅" if diff > 1 else (" ❌" if diff < -1 else "")
        print(f"{tk:<10}{a_no:>7.1f}%{a_yes:>7.1f}%{diff:>+7.1f}%│{nd_no:>10.1f}%{nd_yes:>10.1f}%{n:>5}{mark}")
    except Exception as e:
        print(f"{tk}: {e}")
print("-" * 62)
mno, myes = np.mean(sum_no), np.mean(sum_yes)
mnd_no, mnd_yes = np.nanmean(sum_nd_no), np.nanmean(sum_nd_yes)
print(f"{'全測試集':<10}{mno:>7.1f}%{myes:>7.1f}%{myes-mno:>+7.1f}%│"
      f"{mnd_no:>10.1f}%{mnd_yes:>10.1f}% (有新聞當天)")
print()
print(f"【全測試集】有新聞 {myes:.1f}% vs 無新聞 {mno:.1f}% → "
      f"{'提升，值得留 ✅' if myes-mno > 1 else ('沒提升/變差 ❌' if myes-mno < 0.5 else '雜訊內 ⚪')}")
print(f"【只看有新聞當天】有新聞特徵 {mnd_yes:.1f}% vs 無 {mnd_no:.1f}% → "
      f"{'新聞日有用！可選擇性使用 ✅' if mnd_yes-mnd_no > 2 else '即使在新聞日也沒幫助 ❌'}")
