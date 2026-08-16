# -*- coding: utf-8 -*-
"""
股债配置（资本配置线）—— 参考 Ilmanen 框架的最基础一层
核心问题：价值策略(夏普0.36、回撤-27.85%观测值/-38%真实中枢) 加多少比例的无风险资产(货币基金~2%/年)，
才能把组合回撤压到 ≤15%？代价是多少收益？
关键洞察：加无风险资产只缩放收益和风险，【夏普不变】——这是策略的固有属性，配置改不了。
"""
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126
RF_ANNUAL = 0.02     # 货币基金年化 2%


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
s = pd.Series(rets)


def mdd(x):
    cum = (1 + x).cumprod()
    return (cum / cum.cummax() - 1).min() * 100


rf_monthly = RF_ANNUAL / 12

print("=" * 88)
print(f"股债配置（策略 + 货币基金{int(RF_ANNUAL*100)}%/年），2015-2026")
print("=" * 88)
print(f"{'策略仓位':<10}{'年化':>9}{'超额(超货基)':>12}{'夏普':>7}{'最大回撤':>10}")
print("-" * 88)

rows = []
for x in [1.0, 0.8, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2]:
    c = x * s + (1 - x) * rf_monthly
    annual = ((1 + c).prod() ** (12 / len(c)) - 1) * 100
    sharpe = (c.mean() - rf_monthly) / c.std() * np.sqrt(12) if c.std() > 0 else 0
    dd = mdd(c)
    rows.append((x, annual, sharpe, dd))
    mark = "  <- 回撤≤15%" if dd >= -15 else ""
    print(f"{int(x*100):>7}%{annual:>+8.2f}%{annual-RF_ANNUAL*100:>+11.2f}pp{sharpe:>7.2f}{dd:>+9.2f}%{mark}")

print("-" * 88)
print("注：夏普随仓位几乎不变（约0.36）——加货基只缩放收益和风险，不改善风险调整后收益。")

# 图：资本配置线
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
xs = [r[0] for r in rows]
dd_ = [r[3] for r in rows]
ann = [r[1] for r in rows]
axes[0].plot(xs, dd_, marker='o', linewidth=2, color='crimson')
axes[0].axhline(-15, color='black', linestyle='--', linewidth=1, label='目标回撤 -15%')
axes[0].set_xlabel('策略仓位')
axes[0].set_ylabel('最大回撤 (%)')
axes[0].set_title('仓位 → 最大回撤')
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(dd_, ann, marker='o', linewidth=2, color='steelblue')
axes[1].axvline(-15, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('最大回撤 (%)')
axes[1].set_ylabel('年化收益 (%)')
axes[1].set_title('回撤-收益权衡（左=更稳，右=更高收益）')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/allocation.png", dpi=150)
print("\n[OK] 图表已保存 figures/allocation.png")
