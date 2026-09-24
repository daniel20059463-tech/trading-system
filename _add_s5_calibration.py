"""一次性：為現有分位數模型補「短線 σ5」的 conformal Q（存回 ckpt）。
未來重訓會自動算（train_quantile 已加），此腳本只為現有模型補齊。"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import torch
import config as cfg
import quantile_forecast as qf

for tk in cfg.ALL_STOCKS:
    try:
        q80s = qf._compute_conformal_q_short(tk, alpha=0.2, on="full")
        p = os.path.join(cfg.CHECKPOINT_DIR, f"{tk}_quantile_best.pt")
        ckpt = torch.load(p, map_location="cpu")
        ckpt["conformal_q80_s5"] = q80s
        torch.save(ckpt, p)
        print(f"  {tk:<10} conformal_q80_s5 = {q80s:.3f} ✅")
    except Exception as e:
        print(f"  {tk}: {str(e)[:60]}")
print("完成")
