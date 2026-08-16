# -*- coding: utf-8 -*-
"""
趋势跟踪（时间序列动量 TSMOM）检验 —— 参考 Nick Baltas《Why Trend Following Wins in Chaos》
核心问题：A股的时间序列动量是否 ① 有正收益 ② 与价值策略负相关 ③ 组合后能降回撤？
（注意：区别于之前测的"横截面动量"——A股横截面是反转；这里是每只股票自己的趋势。）

趋势信号：mom12 = 近12月收益（跳过最近1月，避免1月反转污染）
  - TSMOM-broad ：持有 mom12>0 的股票（等权，现金替代下跌趋势股）
  - TSMOM-top20 ：mom12 最高的20只
价值策略：PB+EP+换手率+涨停次数，top20，剔小30%
组合：50% 价值 + 50% 趋势，看回撤/夏普是否改善。
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
WARMUP = 252          # 12个月动量暖机


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

# 12个月动量（跳过最近1月）
mom12 = close.shift(21) / close.shift(252) - 1


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def score_value(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    return pbep + t + lc


def value_returns():
    rets, prev = [], set()
    for i in rebal:
        sc = score_value(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
            prev = set(top)
    return pd.Series(rets, index=pd.Index([dates[i] for i in rebal]))


def trend_returns(mode="broad"):
    rets = []
    for i in rebal:
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        m = mom12.iloc[i]
        valid = m.dropna().index.intersection(universe_ex_small(i))
        if mode == "broad":
            picks = m[valid][m[valid] > 0].index
            if len(picks) < N_HOLD:
                rets.append(0.0)   # 上涨趋势股太少→现金
            else:
                rets.append(fwd[picks].mean())
        else:  # top20
            if len(valid) < N_HOLD:
                rets.append(0.0)
            else:
                top = m[valid].nlargest(N_HOLD).index
                rets.append(fwd[top].mean())
    return pd.Series(rets, index=pd.Index([dates[i] for i in rebal]))


def stats(s):
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, cum


print("=" * 88)
print("趋势跟踪（TSMOM）vs 价值策略，2015-2026 含成本")
print("=" * 88)

v = value_returns()
tb = trend_returns("broad")
tt = trend_returns("top20")

print(f"{'策略':<16}{'年化':>9}{'夏普':>7}{'最大回撤':>10}")
print("-" * 88)
for name, s in [("价值", v), ("趋势-broad", tb), ("趋势-top20", tt)]:
    a, sh, dd, _ = stats(s)
    print(f"{name:<16}{a:>+8.2f}%{sh:>7.2f}{dd:>+9.2f}%")

print("-" * 88)
# 相关性（月度收益）
common = v.index.intersection(tb.index)
vv, ttb = v[common], tb[common]
corr_vt = vv.corr(ttb)
print(f"价值 vs 趋势-broad 月度收益相关系数 = {corr_vt:+.3f}  （<0 = 负相关，可分散）")

# 组合 50/50
combo = 0.5 * vv + 0.5 * ttb
a, sh, dd, cum_combo = stats(combo)
a_v, sh_v, dd_v, _ = stats(vv)
print("\n50%价值 + 50%趋势-broad 组合:")
print(f"  年化 {a:+.2f}% (价值单独 {a_v:+.2f}%)  |  夏普 {sh:.2f} (价值 {sh_v:.2f})  |  回撤 {dd:.2f}% (价值 {dd_v:.2f}%)")

# 图：净值对比
fig, ax = plt.subplots(figsize=(12, 6))
for name, s in [("价值", v), ("趋势-broad", tb), ("50/50组合", combo)]:
    ax.plot((1 + s).cumprod().values, linewidth=2, label=name)
ax.axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
ax.set_title("净值：价值 vs 趋势跟踪 vs 组合（含成本）")
ax.set_ylabel("累计净值")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/trend_following.png", dpi=150)
print("\n[OK] 图表已保存 figures/trend_following.png")
