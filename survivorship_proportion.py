# -*- coding: utf-8 -*-
"""按真实退市占比（~5%）抽样退市股，重估幸存者偏差的真实影响"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
DELISTED_CACHE = "delisted_cache.pkl"
LOOKBACK, REV, HOLD, N_HOLD = 126, 21, 21, 50
ROUND_TRIP = 0.00025 + 0.001 + 0.00025 + 0.0005 + 0.001


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def run(close_df, turn_df, delist_loss):
    daily_ret = close_df.pct_change(fill_method=None)
    dates = close_df.index
    rets, picks = [], 0
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
            prev_held = set(top)
        i += HOLD
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    return (cum.iloc[-1] - 1) * 100, (cum.iloc[-1] ** (12 / len(s)) - 1) * 100, picks


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)

close_l = listed['close']
turn_l = listed['turn']
del_codes = list(delisted['close'].columns)

print("=" * 70)
print("按不同退市股抽样比例重估（退市损失按 -90% 计）")
print("=" * 70)
print(f"{'退市股抽样':<18}{'退市占比':>10}{'累计收益':>10}{'年化收益':>10}{'踩雷':>6}")
print("-" * 70)

for step, label in [(1, "全部254只"), (4, "每4取1(~64只)"), (8, "每8取1(~32只)")]:
    sample = del_codes[::step]
    cd = delisted['close'][sample]
    td = delisted['turn'][sample]
    close_all = pd.concat([close_l, cd], axis=1).reindex(close_l.index)
    turn_all = pd.concat([turn_l, td], axis=1).reindex(turn_l.index)
    ratio = len(sample) / (close_l.shape[1] + len(sample)) * 100
    total, annual, picks = run(close_all, turn_all, -0.9)
    print(f"{label:<18}{ratio:>9.1f}%{total:>+9.2f}%{annual:>+9.2f}%{picks:>6}")

print("-" * 70)
print("参考：无退市股 累计 +119.20%，年化 +7.64%")
print("真实全市场退市占比约 4.8%（254 / 5260）")
