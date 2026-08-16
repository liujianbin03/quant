# -*- coding: utf-8 -*-
"""
低风险异象（low-vol anomaly）检验 —— 参考 Ilmanen《Expected Returns》
低风险异象：低波动/低beta股票的风险调整后收益更高（CAPM预测反了），且回撤更低。
这是五大因子溢价(value/momentum/carry/low-vol/quality)里我们唯一没做过「独立策略」回测的。
问题：A股低波动策略是否 ① 有正收益 ② 回撤低于价值 ③ 与价值负相关可组合降回撤？
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


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def score_value(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    return pbep + t + lc


def score_lowvol(i):
    vol = daily_ret.iloc[i - VOL_WIN:i].std()
    return -zscore(winsorize(vol))   # 低波动好


def backtest(score_fn):
    rets, prev = [], set()
    for i in rebal:
        sc = score_fn(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
            prev = set(top)
    return pd.Series(rets, index=pd.Index([dates[i] for i in rebal]))


def stats(s):
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, cum


v = backtest(score_value)
lv = backtest(score_lowvol)

print("=" * 84)
print("低风险异象：低波动策略 vs 价值策略，2015-2026 含成本")
print("=" * 84)
print(f"{'策略':<16}{'年化':>9}{'夏普':>7}{'最大回撤':>10}")
print("-" * 84)
a_v, sh_v, dd_v, cum_v = stats(v)
a_lv, sh_lv, dd_lv, cum_lv = stats(lv)
print(f"{'价值':<16}{a_v:>+8.2f}%{sh_v:>7.2f}{dd_v:>+9.2f}%")
print(f"{'低波动':<16}{a_lv:>+8.2f}%{sh_lv:>7.2f}{dd_lv:>+9.2f}%")

# 相关性 + 组合
common = v.index.intersection(lv.index)
vv, ll = v[common], lv[common]
corr = vv.corr(ll)
print("-" * 84)
print(f"价值 vs 低波动 相关系数 = {corr:+.3f}")

for w in [0.5, 0.7]:
    combo = w * vv + (1 - w) * ll
    a, sh, dd, _ = stats(combo)
    print(f"{int(w*100)}%价值+{int((1-w)*100)}%低波动: 年化{a:+.2f}% 夏普{sh:.2f} 回撤{dd:.2f}%")

# 图
fig, ax = plt.subplots(figsize=(12, 6))
for name, s in [("价值", v), ("低波动", lv), ("50/50", 0.5*vv+0.5*ll)]:
    ax.plot((1 + s).cumprod().values, linewidth=2, label=name)
ax.axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
ax.set_title("净值：价值 vs 低波动 vs 组合（含成本）")
ax.set_ylabel("累计净值")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/low_risk.png", dpi=150)
print("\n[OK] 图表已保存 figures/low_risk.png")
