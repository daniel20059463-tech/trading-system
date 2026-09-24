"""實驗：隔夜美股(費半SOX)當第20特徵，量隔日方向準確率是否提升。
重點看「大動作日」(|當日漲跌|>3%)——若 SOX 能抓 risk-off/反轉日，這裡才該贏。
公平比較：同種子，唯一差別就是多/少 sox_overnight 這欄。不動生產程式。"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn, yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import config as cfg
from models.lstm_model import get_model
from models.train import FEATURE_COLS

DEVICE = torch.device("cpu")
W = cfg.WINDOW_SIZE
SEEDS = [0, 1, 2]
EPOCHS = 80
BIG = 3.0   # |隔日漲跌|>3% 視為大動作日


def _norm(s):
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize().astype("datetime64[ns]")


# 抓 SOX 一次，建「隔夜對齊」函式
_sx = yf.download("^SOX", start="2023-01-01", progress=False)["Close"].squeeze()
_sxdf = pd.DataFrame({"us_date": _sx.pct_change().index, "sox": (_sx.pct_change() * 100).values}).dropna()
_sxdf["us_date"] = _norm(_sxdf["us_date"])
_sxdf = _sxdf.sort_values("us_date")


def _sox_for(dates):
    """台股各日 ← 嚴格早於該日的最近美股隔夜漲跌；無則 0。"""
    tw = pd.DataFrame({"tw_date": _norm(pd.Series(list(dates)))})
    m = pd.merge_asof(tw, _sxdf, left_on="tw_date", right_on="us_date",
                      direction="backward", allow_exact_matches=False)
    return m["sox"].fillna(0.0).values


def _prep(ticker, use_sox):
    df = pd.read_csv(f"data/raw/{ticker}.csv", index_col=0, parse_dates=True)
    feats = [c for c in FEATURE_COLS if c in df.columns]
    df = df[feats].copy()
    df["sox_overnight"] = _sox_for(df.index)
    feat_cols = feats + (["sox_overnight"] if use_sox else [])
    sub = df[feat_cols].dropna()
    data = sub.values.astype(np.float32)
    tgt = feat_cols.index("pct_change")

    split = int(len(data) * cfg.TRAIN_RATIO)
    scaler = MinMaxScaler().fit(data[:split])
    ds = scaler.transform(data)

    X, yreal = [], []
    for i in range(len(ds) - W):
        X.append(ds[i:i+W]); yreal.append(data[i+W, tgt])
    X = np.array(X, np.float32); yreal = np.array(yreal, np.float32)
    sp = int(len(X) * cfg.TRAIN_RATIO)
    ymin, ymax = yreal[:sp].min(), yreal[:sp].max()
    ys = ((yreal - ymin) / (ymax - ymin + 1e-9)).reshape(-1, 1)
    return X[:sp], ys[:sp], X[sp:], yreal[sp:], (ymin, ymax)


def _train_eval(ticker, use_sox, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte_real, (ymin, ymax) = _prep(ticker, use_sox)
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
    big = np.abs(yte_real) > BIG
    acc_big = hit[big].mean() * 100 if big.sum() else np.nan
    return acc, acc_big, int(big.sum())


print(f"隔夜美股SOX特徵實驗（{len(SEEDS)} 種子平均）")
print(f"{'股':<8}{'無SOX':>8}{'有SOX':>8}{'差異':>8}│{'大動作日:無':>12}{'大動作日:有':>12}{'n':>5}")
print("-" * 64)
sno, syes, ndno, ndyes = [], [], [], []
for tk in cfg.ALL_STOCKS:
    try:
        r_no = [_train_eval(tk, False, s) for s in SEEDS]
        r_yes = [_train_eval(tk, True, s) for s in SEEDS]
        a_no = np.mean([r[0] for r in r_no]); a_yes = np.mean([r[0] for r in r_yes])
        b_no = np.nanmean([r[1] for r in r_no]); b_yes = np.nanmean([r[1] for r in r_yes])
        n = r_yes[0][2]
        sno.append(a_no); syes.append(a_yes); ndno.append(b_no); ndyes.append(b_yes)
        d = a_yes - a_no
        mk = " ✅" if d > 1 else (" ❌" if d < -1 else "")
        print(f"{cfg.ALL_STOCKS.get(tk,''):<7}{a_no:>7.1f}%{a_yes:>7.1f}%{d:>+7.1f}%│{b_no:>11.1f}%{b_yes:>11.1f}%{n:>5}{mk}")
    except Exception as e:
        print(f"{tk}: {str(e)[:50]}")
print("-" * 64)
mno, myes = np.mean(sno), np.mean(syes)
mbno, mbyes = np.nanmean(ndno), np.nanmean(ndyes)
print(f"{'平均':<7}{mno:>7.1f}%{myes:>7.1f}%{myes-mno:>+7.1f}%│{mbno:>11.1f}%{mbyes:>11.1f}%")
print()
print(f"【全測試集】有SOX {myes:.1f}% vs 無 {mno:.1f}% → "
      f"{'提升，值得留 ✅' if myes-mno > 1 else ('沒提升/變差 ❌' if myes-mno < 0.5 else '雜訊內 ⚪')}")
print(f"【大動作日】有SOX {mbyes:.1f}% vs 無 {mbno:.1f}% → "
      f"{'能抓劇烈日！有價值 ✅' if mbyes-mbno > 2 else '劇烈日也沒幫助 ❌'}")
