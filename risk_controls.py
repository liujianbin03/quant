# -*- coding: utf-8 -*-
"""
优先级3：风控"减法" —— 行业集中度约束，砍回撤且几乎不损收益
问题：当前 top20 深价值股高度集中在「银行」单一行业（4-5只），行业踩雷=组合踩雷。
方法：行业上限 = 每个证监会门类最多持 3 只，超出的让位给次优股票。
对比：top20 等权(基准) vs top20 行业上限3只，看回撤/夏普是否改善。

口径：PB+EP+换手率，top20，剔小30%，月度含成本。
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
industry = si['industry'].reindex(close.columns)
ind_letter = industry.str.slice(0, 1).fillna('Z')   # 门类首字母

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


def select(i, max_per_ind=None):
    """选 top20；max_per_ind=None 无行业约束，否则每个门类最多 max_per_ind 只"""
    sc = score_strategy(i)
    valid = sc.dropna().index.intersection(universe_ex_small(i))
    s = sc[valid].sort_values(ascending=False)
    if max_per_ind is None:
        return s.head(N_HOLD)
    picked, cnt = [], {}
    for code in s.index:
        L = ind_letter[code]
        if cnt.get(L, 0) >= max_per_ind:
            continue
        picked.append(code)
        cnt[L] = cnt.get(L, 0) + 1
        if len(picked) == N_HOLD:
            break
    return s[picked]


def backtest(max_per_ind=None):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal:
        top = select(i, max_per_ind)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = universe_ex_small(i)
        bench.append(fwd[v].mean())
        if len(top) >= N_HOLD:
            gross = fwd[top.index].mean()
            turnv = 1 - len(set(top.index) & prev) / N_HOLD if prev else 1.0
            rets.append(gross - turnv * ROUND_TRIP)
            turns.append(turnv)
            prev = set(top.index)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, annual - b_annual, sharpe, mdd, np.mean(turns) * 100, cum


print("=" * 88)
print("行业集中度约束：PB+EP+换手率 top20 剔小30%（含成本）")
print("=" * 88)
print(f"{'约束':<14}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}{'换手':>7}")
print("-" * 88)
r0 = backtest(None)
print(f"{'无约束(基准)':<14}{r0[0]:>+8.2f}%{r0[1]:>+7.2f}{r0[2]:>7.2f}{r0[3]:>7.2f}%{r0[4]:>6.1f}%")
for cap in [5, 4, 3, 2]:
    r = backtest(cap)
    print(f"{'每行业≤'+str(cap)+'只':<14}{r[0]:>+8.2f}%{r[1]:>+7.2f}{r[2]:>7.2f}{r[3]:>7.2f}%{r[4]:>6.1f}%")

# 当前 top20 的行业分布
print("\n当前 top20 行业分布（门类）：")
top_now = select(len(dates) - 1, None)
dist = ind_letter[top_now.index].value_counts()
for L, c in dist.items():
    print(f"  {L}类: {c} 只")
