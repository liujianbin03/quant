# -*- coding: utf-8 -*-
"""
可执行性评估：PB+EP + 剔小30% 的 top10 vs top50（预期收益/回撤/换手）
—— 研究版用 top50，真人实盘更现实是 top10：头部 alpha 更集中，但要验证
    集中持仓是否显著恶化回撤/方差。含成本，对比全区间 + 样本外(2021+)。
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"

HOLD = 21
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30


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


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)

close = listed['close']
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
split_pos = close.index.get_indexer([pd.Timestamp("2021-01-04")], method='nearest')[0]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def score_pb_ep(i):
    return -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))


def universe_ex_small(i):
    mi = mcap.iloc[i]
    thr = mi.quantile(SMALL_PCT)
    return mi[mi > thr].index


def backtest(rebal_list, n_hold):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal_list:
        sc = score_pb_ep(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index.intersection(universe_ex_small(i))
        bench.append(fwd[valid].mean())
        if len(valid) >= n_hold:
            top = sc[valid].nlargest(n_hold).index
            gross = fwd[top].mean()
            turn = 1 - len(set(top) & prev) / n_hold if prev else 1.0
            rets.append(gross - turn * ROUND_TRIP)
            turns.append(turn)
            prev = set(top)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, annual - b_annual, sharpe, mdd, np.mean(turns) * 100, cum


in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]

print("=" * 92)
print("PB+EP + 剔小30%：持仓数量对比（含成本）")
print("=" * 92)
print(f"{'持仓':<8}{'区间':<14}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}{'换手':>7}")
print("-" * 92)

for n in [10, 20, 50]:
    r_full = backtest(rebal, n)
    r_out = backtest(out_rebal, n)
    print(f"top{n:<5}{'全区间2015-26':<14}{r_full[0]:>+8.2f}%{r_full[1]:>+7.2f}{r_full[2]:>7.2f}"
          f"{r_full[3]:>7.2f}%{r_full[4]:>6.1f}%")
    print(f"{'':<8}{'样本外2021-26':<14}{r_out[0]:>+8.2f}%{r_out[1]:>+7.2f}{r_out[2]:>7.2f}"
          f"{r_out[3]:>7.2f}%{r_out[4]:>6.1f}%")
    print("-" * 92)

# 逐年收益（top10 vs top50，看 top10 是否放大单年波动）
print("\ntop10 vs top50 逐年收益（%）")
print(f"{'年份':<8}{'top10':>10}{'top50':>10}")
years = {}
for i in rebal:
    y = dates[i].year
    if y not in years:
        years[y] = []
    years[y].append(i)

for y, idxs in sorted(years.items()):
    if len(idxs) < 11:
        continue
    s10 = pd.Series([score_pb_ep(i) for i in idxs])
    r10, r50 = [], []
    for i in idxs:
        sc = score_pb_ep(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index.intersection(universe_ex_small(i))
        if len(valid) >= 50:
            r10.append(fwd[sc[valid].nlargest(10).index].mean())
            r50.append(fwd[sc[valid].nlargest(50).index].mean())
    if r10:
        a10 = ((1 + pd.Series(r10)).prod() - 1) * 100
        a50 = ((1 + pd.Series(r50)).prod() - 1) * 100
        print(f"{y:<8}{a10:>+9.1f}%{a50:>+9.1f}%")
