# -*- coding: utf-8 -*-
"""
风险平价加权：top20 内按 1/波动率 配权（替代等权），看能否降低回撤、提升夏普
对比：等权(基准) vs 风险平价(1/vol)。均满仓、不加杠杆、含成本、剔小30%。
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126
VOL_WIN = 60


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


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
daily_ret = close.pct_change(fill_method=None)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]


def score_strategy(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    return pbep + t + lc


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def backtest(weighting="equal"):
    rets, prev_w = [], {}
    for i in rebal:
        sc = score_strategy(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) < N_HOLD:
            continue
        top = sc[v].nlargest(N_HOLD).index
        if weighting == "equal":
            w = pd.Series(1.0 / N_HOLD, index=top)
        else:  # 风险平价：1/vol
            vol = daily_ret.iloc[i - VOL_WIN:i].std()[top]
            vol = vol.replace(0, np.nan).fillna(vol.median())
            inv = 1.0 / vol
            w = inv / inv.sum()
        gross = (fwd[top] * w).sum()
        # 换手（按权重重叠近似）
        if len(prev_w):
            common = w.index.intersection(prev_w.index)
            stay = min(w[common].sum(), prev_w[common].sum())
            turnv = 1 - stay
        else:
            turnv = 1.0
        rets.append(gross - turnv * ROUND_TRIP)
        prev_w = w
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, cum


print("=" * 72)
print("风险平价 vs 等权（PB+EP+换手率+涨停次数 top20 剔小30%，含成本）")
print("=" * 72)
print(f"{'加权方式':<12}{'年化':>9}{'夏普':>7}{'最大回撤':>10}")
print("-" * 72)
for wgt in ["equal", "riskparity"]:
    a, sh, dd, cum = backtest(wgt)
    label = "等权" if wgt == "equal" else "风险平价"
    print(f"{label:<12}{a:>+8.2f}%{sh:>7.2f}{dd:>+9.2f}%")
print("-" * 72)
print("目标：回撤≤15%、夏普≥0.4")
