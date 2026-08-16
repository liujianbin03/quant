# -*- coding: utf-8 -*-
"""
诚实回答"3个月能赚多少"：算策略的 3 个月滚动收益分布，而不是给一个点估计
策略 = PB+EP+换手率，top20，剔小30%，含成本，月度。
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
INDEX_NAME_CACHE = "index_name_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126


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
with open(INDEX_NAME_CACHE, 'rb') as f:
    iname = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
hs300 = iname['index'].reindex(close.index)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]


def score_strategy(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    return pbep + t


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


# 逐月策略收益
monthly, prev = [], set()
for i in rebal:
    sc = score_strategy(i)
    fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
    v = sc.dropna().index.intersection(universe_ex_small(i))
    if len(v) >= N_HOLD:
        top = sc[v].nlargest(N_HOLD).index
        turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
        monthly.append(fwd[top].mean() - turnv * ROUND_TRIP)
        prev = set(top)

m = pd.Series(monthly)

# 3个月滚动（每3个连续月度）
m3 = (1 + m).rolling(3).apply(np.prod, raw=True) - 1
m3 = m3.dropna()

# 沪深300 3个月滚动（月度采样对齐）
hs_monthly = pd.Series([hs300.iloc[i + HOLD] / hs300.iloc[i] - 1 for i in rebal], index=m.index[:len(rebal)])
hs3 = (1 + hs_monthly).rolling(3).apply(np.prod, raw=True) - 1
hs3 = hs3.dropna()


def stat(name, s):
    print(f"{name:<16} 均值{s.mean()*100:>+6.2f}%  中位{s.median()*100:>+6.2f}%  "
          f"标准差{s.std()*100:>5.2f}%  最差{s.min()*100:>+6.2f}%  最好{s.max()*100:>+6.2f}%  "
          f"正收益占比{(s>0).mean()*100:>4.0f}%")


print("=" * 92)
print("3 个月滚动收益分布（策略 vs 沪深300，历史 2015-2026 回测）")
print("=" * 92)
stat("策略(含成本)", m3)
stat("沪深300", hs3)
print("-" * 92)
print(f"策略 3 个月：亏>5% 概率 {(m3 < -0.05).mean()*100:.0f}%  |  赚>5% 概率 {(m3 > 0.05).mean()*100:.0f}%")
print(f"沪深300 3个月：亏>5% 概率 {(hs3 < -0.05).mean()*100:.0f}%  |  赚>5% 概率 {(hs3 > 0.05).mean()*100:.0f}%")
