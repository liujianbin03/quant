# -*- coding: utf-8 -*-
"""
Monte Carlo 鲁棒性检验 —— 参考 Kevin Davey《It's All About Process》
用 bootstrap（有放回重采样月度收益 10000 次）测价值策略的 Sharpe / 回撤对"收益序列具体形态"的敏感性：
  - 若 95% 置信区间跨过 0，说明这个夏普可能是运气
  - 回撤分布的宽度，说明"历史 -27.85%"到底有多不确定
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
N_BOOT = 10000


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


# 策略月度收益
rets, prev = [], set()
for i in rebal:
    sc = score_strategy(i)
    fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
    v = sc.dropna().index.intersection(universe_ex_small(i))
    if len(v) >= N_HOLD:
        top = sc[v].nlargest(N_HOLD).index
        turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
        rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
        prev = set(top)

r = np.array(rets)
n = len(r)

# 实际值
def sharpe_of(x):
    return x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0

def mdd_of(x):
    cum = np.cumprod(1 + x)
    return (cum / np.maximum.accumulate(cum) - 1).min() * 100

sh_real, dd_real = sharpe_of(r), mdd_of(r)

# Bootstrap
rng = np.random.default_rng(42)
sh_boot, dd_boot = [], []
for _ in range(N_BOOT):
    x = r[rng.integers(0, n, n)]   # 有放回重采样
    sh_boot.append(sharpe_of(x))
    dd_boot.append(mdd_of(x))
sh_boot = np.array(sh_boot)
dd_boot = np.array(dd_boot)

print("=" * 72)
print(f"Monte Carlo 鲁棒性检验（bootstrap {N_BOOT} 次）")
print("=" * 72)
print(f"实际月收益样本: {n} 期")
print(f"  实际夏普 {sh_real:.2f}  |  实际最大回撤 {dd_real:.2f}%")
print(f"\n夏普 bootstrap 分布:")
print(f"  中位 {np.median(sh_boot):.2f}  |  95%区间 [{np.percentile(sh_boot,2.5):.2f}, {np.percentile(sh_boot,97.5):.2f}]")
print(f"  P(夏普<0) = {(sh_boot<0).mean()*100:.1f}%   <- 越低越说明不是运气")
print(f"\n最大回撤 bootstrap 分布:")
print(f"  中位 {np.median(dd_boot):.2f}%  |  95%区间 [{np.percentile(dd_boot,2.5):.2f}%, {np.percentile(dd_boot,97.5):.2f}%]")
print(f"  P(回撤<-30%) = {(dd_boot<-30).mean()*100:.1f}%   <- 深回撤发生的概率")
