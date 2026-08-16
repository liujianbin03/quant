# -*- coding: utf-8 -*-
"""
carry（股息率）检验 —— 五因子里最后一个未测的
股息率 = 上一财年每股派息(税前) / 当前价（约4个月年报滞后，与项目估值滞后口径一致）
问题：A股股息率是否 ① 有正收益 ② 与价值负/正交 ③ 组合能降回撤？
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
DIV_CACHE = "dividend_cache.pkl"

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
with open(DIV_CACHE, 'rb') as f:
    dps = pickle.load(f)

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


def score_carry(i):
    """股息率（高好）。上一财年派息 / 现价，4个月年报滞后近似"""
    y = dates[i].year - 1
    if y < int(dps.index.min()):
        return pd.Series(np.nan, index=close.columns)
    row = dps.loc[y] if y in dps.index else pd.Series(np.nan, index=close.columns)
    row = row.reindex(close.columns)
    div_yield = row / close.iloc[i]
    return zscore(winsorize(div_yield))


def backtest(score_fn):
    rets, prev, ret_dates = [], set(), []
    for i in rebal:
        sc = score_fn(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
            prev = set(top)
            ret_dates.append(dates[i])
    return pd.Series(rets, index=pd.Index(ret_dates))


def stats(s):
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, cum


v = backtest(score_value)
c = backtest(score_carry)

print("=" * 84)
print("carry（股息率）vs 价值策略，2015-2026 含成本")
print("=" * 84)
print(f"{'策略':<16}{'年化':>9}{'夏普':>7}{'最大回撤':>10}")
print("-" * 84)
a_v, sh_v, dd_v, cum_v = stats(v)
a_c, sh_c, dd_c, cum_c = stats(c)
print(f"{'价值':<16}{a_v:>+8.2f}%{sh_v:>7.2f}{dd_v:>+9.2f}%")
print(f"{'股息率':<16}{a_c:>+8.2f}%{sh_c:>7.2f}{dd_c:>+9.2f}%")

common = v.index.intersection(c.index)
vv, cc = v[common], c[common]
corr = vv.corr(cc)
print("-" * 84)
print(f"价值 vs 股息率 相关系数 = {corr:+.3f}")

for w in [0.5, 0.7]:
    combo = w * vv + (1 - w) * cc
    a, sh, dd, _ = stats(combo)
    print(f"{int(w*100)}%价值+{int((1-w)*100)}%股息率: 年化{a:+.2f}% 夏普{sh:.2f} 回撤{dd:.2f}%")

fig, ax = plt.subplots(figsize=(12, 6))
for name, s in [("价值", v), ("股息率", c), ("50/50", 0.5*vv+0.5*cc)]:
    ax.plot((1 + s).cumprod().values, linewidth=2, label=name)
ax.axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
ax.set_title("净值：价值 vs 股息率 vs 组合（含成本）")
ax.set_ylabel("累计净值")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/carry.png", dpi=150)
print("\n[OK] 图表已保存 figures/carry.png")
