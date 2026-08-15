# -*- coding: utf-8 -*-
"""
选股清单：输出 PB+EP+换手率+涨停次数 + 剔小30% 策略的当前 top-N 买入清单 + 与沪深300对比
用法：
    python signal_picker.py          # 默认 top20
    python signal_picker.py 50       # 自定义持仓数

输出：可读的「现在该买哪几只」清单（代码+名称+现价+PB+EP%+分数），
     以及策略近12个月 vs 沪深300 的表现。
"""
import sys
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
INDEX_NAME_CACHE = "index_name_cache.pkl"

HOLD = 21
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
N_HOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 20


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
with open(INDEX_NAME_CACHE, 'rb') as f:
    iname = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
names = iname['name'].reindex(close.columns)
hs300 = iname['index'].reindex(close.index)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
daily_ret = close.pct_change(fill_method=None)
dates = close.index


def score_strategy(i):
    """PB+EP + 换手率(低) + 涨停次数(少) —— 参数搜索样本外最优"""
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    return pbep + t + lc


def universe_ex_small(i):
    mi = mcap.iloc[i]
    thr = mi.quantile(SMALL_PCT)
    return mi[mi > thr].index


# ============ 当前 top-N ============
i = len(dates) - 1
cur_date = dates[i]
sc = score_strategy(i)
valid = sc.dropna().index.intersection(universe_ex_small(i))
top = sc[valid].nlargest(N_HOLD)

print("=" * 88)
print(f"A股价值策略选股清单（PB+EP+换手率+涨停次数 + 剔小30%）  数据截至 {cur_date.date()}  持仓 top{N_HOLD}")
print("=" * 88)
print(f"{'排名':<5}{'代码':<12}{'名称':<12}{'收盘价':>9}{'PB':>8}{'EP%':>8}{'分数':>8}")
print("-" * 88)
for rank, code in enumerate(top.index, 1):
    nm = names[code] if pd.notna(names[code]) else '?'
    px = close.iloc[i][code]
    pbv = pb.iloc[i][code]
    epv = ep.iloc[i][code] * 100 if pd.notna(ep.iloc[i][code]) else np.nan
    print(f"{rank:<5}{code:<12}{nm:<12}{px:>9.2f}{pbv:>8.2f}{epv:>7.2f}%{top[code]:>+8.2f}")

print("-" * 88)
print(f"候选池（剔小30%后）: {len(valid)} 只，PB 中位 {pb.iloc[i][valid].median():.2f}，"
      f"EP 中位 {ep.iloc[i][valid].median()*100:.2f}%")

# ============ 近12个月表现 vs 沪深300 ============
rebal = [k for k in range(0, len(dates), HOLD) if k + HOLD < len(dates)]
recent = rebal[-12:]
rets = []
prev = set()
for k in recent:
    sck = score_strategy(k)
    fwd = close.iloc[k + HOLD] / close.iloc[k] - 1
    v = sck.dropna().index.intersection(universe_ex_small(k))
    if len(v) >= N_HOLD:
        t = sck[v].nlargest(N_HOLD).index
        turnv = 1 - len(set(t) & prev) / N_HOLD if prev else 1.0
        rets.append(fwd[t].mean() - turnv * ROUND_TRIP)
        prev = set(t)
strat_cum = (1 + pd.Series(rets)).prod() - 1 if rets else 0
t0, t1 = dates[recent[0]], dates[recent[-1] + HOLD]
idx_ret = hs300[t1] / hs300[t0] - 1

print("\n近12个月表现:")
print(f"  策略(含成本): {strat_cum*100:+.2f}%")
print(f"  沪深300:      {idx_ret*100:+.2f}%")
print(f"  超额:         {(strat_cum-idx_ret)*100:+.2f} pp")
print("\n说明：本清单为研究输出，非投资建议。调仓频率=月度，回撤可能达 -40%。")
