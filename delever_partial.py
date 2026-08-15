# -*- coding: utf-8 -*-
"""
降仓机制 v2：替代失败的"永久清仓"，测两种能保住反弹的设计
  A) 回撤部分降仓：DD≤-15% 时仓位降到 50%，DD 修复到 -8% 自动回到满仓（带迟滞防抖）
  B) 波动率目标：仓位 = 目标月波动 / 近6月已实现波动（封顶100%），危机时自动减仓
对照：基线(永不降仓)。
目标：把全样本回撤 -27.85% 压向 ≤15%，且尽量保住夏普≥0.4、正收益。
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


def strategy_returns():
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
        else:
            rets.append(0.0)
    return pd.Series(rets)


def metrics(s):
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, cum


base = strategy_returns()

# A) 回撤部分降仓
def dd_delever(dd_stop=0.15, recover=0.08, cut=0.5):
    out, nav, peak, expo = [], 1.0, 1.0, 1.0
    for r in base:
        if expo < 1.0 and (nav / peak - 1) > -recover:
            expo = 1.0
        r_scaled = r * expo
        out.append(r_scaled)
        nav *= (1 + r_scaled)
        peak = max(peak, nav)
        if (nav / peak - 1) <= -dd_stop:
            expo = cut
    return pd.Series(out)

# B) 波动率目标
def vol_target(target_vol=0.03, lookback=6, cap=1.0):
    out, hist = [], []
    for r in base:
        hist.append(r)
        if len(hist) >= lookback:
            trail = pd.Series(hist[-lookback:]).std()
            expo = min(cap, target_vol / trail) if trail > 0 else cap
        else:
            expo = cap
        out.append(r * expo)
    return pd.Series(out)


print("=" * 84)
print("降仓机制 v2（PB+EP+换手率+涨停次数 top20 剔小30%，含成本）")
print("=" * 84)
print(f"{'机制':<22}{'年化':>9}{'夏普':>7}{'最大回撤':>11}{'平均仓位':>9}")
print("-" * 84)

a, sh, dd, _ = metrics(base)
print(f"{'基线(永不降仓)':<22}{a:>+8.2f}%{sh:>7.2f}{dd:>+10.2f}%{'100%':>9}")

for dd_stop in [0.15, 0.12]:
    s = dd_delever(dd_stop=dd_stop)
    a, sh, dd, _ = metrics(s)
    print(f"{'回撤降仓50%@'+str(int(dd_stop*100))+'%':<22}{a:>+8.2f}%{sh:>7.2f}{dd:>+10.2f}%"
          f"{100*((s.abs()>1e-9).mean()):>8.0f}%")

for tv in [0.04, 0.03, 0.025]:
    s = vol_target(target_vol=tv)
    a, sh, dd, _ = metrics(s)
    expo_avg = float(np.mean([min(1.0, tv / pd.Series(base[max(0,k-6):k]).std()) if k >= 6 else 1.0 for k in range(1, len(base)+1)]))
    print(f"{'波动率目标'+str(int(tv*1000))+'bp':<22}{a:>+8.2f}%{sh:>7.2f}{dd:>+10.2f}%{expo_avg*100:>8.0f}%")

print("-" * 84)
print("修订后目标：回撤≤15%、夏普≥0.4")
