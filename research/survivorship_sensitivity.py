# -*- coding: utf-8 -*-
"""幸存者偏差敏感性：退市损失假设从 -100% 到 -50%，看修正后收益区间"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
DELISTED_CACHE = "delisted_cache.pkl"
LOOKBACK, REV, HOLD, N_HOLD = 126, 21, 21, 50
ROUND_TRIP = 0.00025 + 0.001 + 0.00025 + 0.0005 + 0.001  # 0.30%


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def run(close_df, turn_df, delist_loss):
    daily_ret = close_df.pct_change(fill_method=None)
    dates = close_df.index
    rets, turnovers, picks = [], [], 0
    prev_held = set()
    i = LOOKBACK
    while i + HOLD < len(dates):
        t0, t1, t_rev, t2 = dates[i-LOOKBACK], dates[i], dates[i-REV], dates[i+HOLD]
        rev = close_df.loc[t1] / close_df.loc[t_rev] - 1
        vol = daily_ret.loc[t0:t1].std()
        turn = turn_df.loc[t_rev:t1].mean()
        fwd = close_df.loc[t2] / close_df.loc[t1] - 1
        dmask = close_df.loc[t1].notna() & close_df.loc[t2].isna()
        fwd = fwd.copy()
        fwd[dmask] = delist_loss
        valid = rev.dropna().index.intersection(vol.dropna().index).intersection(turn.dropna().index)
        if len(valid) >= N_HOLD:
            score = -zscore(rev[valid]) - zscore(vol[valid]) - zscore(turn[valid])
            top = score.nlargest(N_HOLD).index
            gross = fwd[top].mean()
            picks += int(dmask[top].sum())
            turnover = 1 - len(set(top) & prev_held) / N_HOLD if prev_held else 1.0
            rets.append(gross - turnover * ROUND_TRIP)
            turnovers.append(turnover)
            prev_held = set(top)
        i += HOLD
    return pd.Series(rets), picks


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)
close_all = pd.concat([listed['close'], delisted['close']], axis=1).reindex(listed['close'].index)
turn_all = pd.concat([listed['turn'], delisted['turn']], axis=1).reindex(listed['turn'].index)

print("=" * 70)
print("退市损失假设敏感性（含全部 254 只退市股，含成本）")
print("=" * 70)
print(f"{'退市损失假设':<16}{'累计收益':>10}{'年化收益':>10}{'踩雷次数':>10}")
print("-" * 70)
for loss in [-1.0, -0.9, -0.8, -0.7, -0.5]:
    s, picks = run(close_all, turn_all, loss)
    cum = (1 + s).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    print(f"退市损失 {loss*100:>5.0f}%    {total:>+9.2f}%{annual:>+9.2f}%{picks:>10}")
print("-" * 70)
print("参考：无退市股（有幸存者偏差）累计 +119.20%，年化 +7.64%")
