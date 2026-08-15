# -*- coding: utf-8 -*-
"""
⑤ PB+低波动 策略的幸存者偏差检验：加入退市股，重估组合 alpha
- 核心风险：低波动因子可能偏爱"僵尸股"（交易清淡、横盘），这类股退市风险高
- 检验：加入退市股后，PB+低波动 是否比纯价值跌得更多？
- 处理：退市股持有期退市 fwd=-90%；按不同占比抽样（全部254/每4/每8）
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
DELISTED_CACHE = "delisted_cache.pkl"
VAL_CACHE = "val_cache.pkl"
DELISTED_VAL_CACHE = "delisted_val_cache.pkl"

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
LOWVOL_WIN = 60


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def clean(f):
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


def winsorize(s, q=0.01):
    s = s.astype(float)
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def run(close_df, pb_df, delist_loss):
    dates = close_df.index
    rets, picks = [], 0
    prev_held = set()
    i = 0
    while i + HOLD < len(dates):
        t1, t2 = dates[i], dates[i + HOLD]
        fwd = close_df.loc[t2] / close_df.loc[t1] - 1
        dmask = close_df.loc[t1].notna() & close_df.loc[t2].isna()
        fwd = fwd.copy()
        fwd[dmask] = delist_loss

        # 低波动：过去60日收益率标准差
        if i < LOWVOL_WIN:
            lowvol = pd.Series(np.nan, index=close_df.columns)
        else:
            window = close_df.iloc[i - LOWVOL_WIN:i]
            lowvol = window.pct_change().iloc[1:].std()

        score = -zscore(clean(pb_df.loc[t1])) + (-zscore(winsorize(lowvol)))
        valid = score.dropna().index
        if len(valid) >= N_HOLD:
            top = score[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            picks += int(dmask[top].sum())
            turnover = 1 - len(set(top) & prev_held) / N_HOLD if prev_held else 1.0
            rets.append(gross - turnover * ROUND_TRIP)
            prev_held = set(top)
        i += HOLD
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    return (cum.iloc[-1] - 1) * 100, (cum.iloc[-1] ** (12 / len(s)) - 1) * 100, picks


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(DELISTED_VAL_CACHE, 'rb') as f:
    dval = pickle.load(f)

close_l = listed['close']
pb_l = val['pb'].reindex(close_l.index)

del_codes = list(delisted['close'].columns)
cd_all = delisted['close']
pb_d = dval['pb'].reindex(cd_all.index)

print("=" * 76)
print("PB+低波动 策略幸存者偏差检验（退市损失 -90%）")
print("=" * 76)
print(f"{'退市股抽样':<16}{'退市占比':>10}{'累计收益':>10}{'年化收益':>10}{'踩雷':>6}")
print("-" * 76)

for step, label in [(1, "全部254只"), (4, "每4取1(~64)"), (8, "每8取1(~32)")]:
    sample = del_codes[::step]
    cd = cd_all[sample]
    pb_dd = pb_d[sample].reindex(close_l.index)

    close_all = pd.concat([close_l, cd], axis=1).reindex(close_l.index)
    pb_all = pd.concat([pb_l, pb_dd], axis=1)

    ratio = len(sample) / (close_l.shape[1] + len(sample)) * 100
    total, annual, picks = run(close_all, pb_all, -0.9)
    print(f"{label:<16}{ratio:>9.1f}%{total:>+9.2f}%{annual:>+9.2f}%{picks:>6}")

print("-" * 76)
print("参考：无退市股 PB+低波动 累计 +385.3%，年化 +15.20%（基线）")
print("参考：纯价值 无退市 +313.74% → 含退市(全部254) 略降但远好于反转")
