# -*- coding: utf-8 -*-
"""
资金流信号（融资余额变化）完整检验：是否与价值正交、是否有 IC、加进价值能否提升
信号：score = -zscore(融资余额环比变化率)，散户加杠杆=过热→低未来收益（反向）
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
MARGIN_CACHE = "margin_cache.pkl"

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


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)
with open(DIV_CACHE, 'rb') as f:
    dps = pickle.load(f)
with open(MARGIN_CACHE, 'rb') as f:
    mdf = pickle.load(f)

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

# 融资数据对齐到 close 的日期（mdf index 是抓取的调仓日）
mdf = mdf.reindex(columns=close.columns)
margin_chg = mdf.pct_change(fill_method=None)

rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]
# 只保留 margin 数据覆盖的调仓日
rebal = [i for i in rebal if dates[i] in mdf.index]


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def carry_yield(i):
    y = dates[i].year - 1 if dates[i].month >= 5 else dates[i].year - 2
    if y < int(dps.index.min()):
        return pd.Series(np.nan, index=close.columns)
    row = dps.loc[y] if y in dps.index else pd.Series(np.nan, index=close.columns)
    return row.reindex(close.columns) / close.iloc[i]


def score_value(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    cy = zscore(winsorize(carry_yield(i))).fillna(0)
    return pbep + t + lc + 0.3 * cy


def score_margin(i):
    d = dates[i]
    chg = margin_chg.loc[d] if d in margin_chg.index else pd.Series(np.nan, index=close.columns)
    return -zscore(winsorize(chg))


# ============ IC + 相关 ============
print("=" * 88)
print("融资余额变化 信号检验（2019-2026，散户杠杆反向）")
print("=" * 88)
ics, corrs = [], []
for i in rebal[1:]:
    sm = score_margin(i)
    fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
    ics.append(spearman(sm, fwd))
    corrs.append(spearman(sm, score_value(i)))
s = pd.Series(ics).dropna()
c = pd.Series(corrs).dropna()
t = abs(s.mean() / s.std() * np.sqrt(len(s))) if s.std() > 0 else 0
print(f"融资信号 IC = {s.mean():+.4f}  |t| {t:.2f}  样本 {len(s)} 期")
print(f"与价值信号截面相关 = {c.mean():+.4f}（<0.3 正交）")


# ============ 回测：价值 vs 价值+融资 ============
def backtest(fn):
    rets, prev, rdates = [], set(), []
    for i in rebal:
        sc = fn(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
            prev = set(top)
            rdates.append(dates[i])
    return pd.Series(rets, index=pd.Index(rdates))


def stats(s):
    if len(s) < 10:
        return 0, 0, 0
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd


print("\n回测（含成本，2019-2026）：")
print(f"{'策略':<20}{'年化':>9}{'夏普':>7}{'回撤':>9}")
print("-" * 88)
for name, fn in [("价值", score_value),
                 ("价值+0.5融资", lambda i: score_value(i) + 0.5 * score_margin(i).fillna(0)),
                 ("价值+1.0融资", lambda i: score_value(i) + 1.0 * score_margin(i).fillna(0))]:
    a, sh, dd = stats(backtest(fn))
    print(f"{name:<20}{a:>+8.2f}%{sh:>7.2f}{dd:>+8.2f}%")
