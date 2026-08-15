# -*- coding: utf-8 -*-
"""
优先级2：多重检验校正 —— 我们试了 N 个因子，多少"显著"其实是运气？
用 Harvey-Liu (2015) Backtesting 的思路：
  - 规则1（Harvey-Liu）：|t| > 3.0 才算显著（不是 2.0）
  - 规则2（Bonferroni）：α=0.05 摊到 N 个因子，双尾临界 t = Φ⁻¹(1-α/(2N))

把所有测过的因子在「同一框架」下重算 IC + |t|，看哪些在校正后还站得住。
口径：月度，分数方向统一为"高=持有"，正 IC=有效；区间 2015-2026。
"""
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"

HOLD = 21
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


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


def tstat(ics):
    s = pd.Series(ics).dropna()
    if len(s) < 5 or s.std() == 0:
        return 0.0
    return s.mean() / s.std() * np.sqrt(len(s))


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
ps = val['ps'].reindex(close.index)
pcf = val['pcf'].reindex(close.index)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

daily_ret = close.pct_change(fill_method=None)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]
nan_s = pd.Series(np.nan, index=close.columns)


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ 因子（分数高=持有） ============
def f_pb(i):
    return -zscore(clean(pb.iloc[i]))


def f_ep(i):
    return zscore(winsorize(ep.iloc[i]))


def f_pbep(i):
    return f_pb(i) + f_ep(i)


def f_turn(i):
    return -zscore(winsorize(turn.iloc[i - 21:i].mean()))


def f_abnturn(i):
    t1 = turn.iloc[i - 21:i].mean()
    t6 = turn.iloc[i - WARMUP:i].mean()
    return -zscore(winsorize(t1 / t6.replace(0, np.nan)))


def f_max(i):
    return -zscore(winsorize(daily_ret.iloc[i - 21:i].max()))


def f_limit(i):
    return -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))


def f_rev1(i):
    return -zscore(winsorize(close.iloc[i] / close.iloc[i - 21] - 1))


def f_mom6(i):
    return zscore(winsorize(close.iloc[i] / close.iloc[i - 126] - 1))


def f_vol(i):
    return -zscore(winsorize(daily_ret.iloc[i - 60:i].std()))


FACTORS = {
    "PB(低好)": f_pb,
    "EP(高好)": f_ep,
    "PB+EP": f_pbep,
    "换手率(低好)": f_turn,
    "异常换手(低好)": f_abnturn,
    "MAX(低好)": f_max,
    "涨停次数(低好)": f_limit,
    "反转1月(低好)": f_rev1,
    "动量6月(高好)": f_mom6,
    "波动率(低好)": f_vol,
}

# ============ 算 IC + t ============
rows = []
for name, fn in FACTORS.items():
    ics = [spearman(fn(i), fwd_ret(i)) for i in rebal]
    s = pd.Series(ics).dropna()
    rows.append((name, s.mean(), tstat(ics)))

N = len(FACTORS)
t_hl = 3.0                          # Harvey-Liu 规则
t_bonf = norm.ppf(1 - 0.05 / (2 * N))   # Bonferroni 双尾临界

print("=" * 88)
print(f"多重检验校正：共试 {N} 个因子（同一框架重算 IC）")
print("=" * 88)
print(f"阈值：Harvey-Liu |t|>3.0 ；Bonferroni |t|>{t_bonf:.2f} (α=0.05/{N})")
print("-" * 88)
print(f"{'因子':<16}{'IC':>9}{'|t|':>8}   {'Harvey-Liu>3':>13}{'Bonferroni':>13}")
print("-" * 88)
for name, ic, t in sorted(rows, key=lambda x: -x[2]):
    ok_hl = "✓" if abs(t) >= t_hl else "✗"
    ok_b = "✓" if abs(t) >= t_bonf else "✗"
    print(f"{name:<16}{ic:>+9.4f}{abs(t):>8.2f}   {ok_hl:>13}{ok_b:>13}")
print("-" * 88)
print("✗ = 可能是多重检验/数据挖掘出的噪声，实盘存疑。")
