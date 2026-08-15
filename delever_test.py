# -*- coding: utf-8 -*-
"""
回撤降仓线测试：8% / 12% / 15% / 20% 哪个合理？对照修订后目标（回撤≤15%、夏普≥0.4）
触发即清仓+冻结（模拟"降仓暂停等待人工确认"），看能否把回撤压进阈值且不过度伤收益。
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


def simulate(dd_stop):
    rets, turns, trigger = [], [], None
    prev = set()
    nav, peak, stopped = 1.0, 1.0, False
    for k, i in enumerate(rebal):
        if stopped:
            rets.append(0.0)
            continue
        sc = score_strategy(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            r = fwd[top].mean() - turnv * ROUND_TRIP
            rets.append(r)
            prev = set(top)
            nav *= (1 + r)
            peak = max(peak, nav)
            if (nav / peak - 1) <= -dd_stop:
                stopped = True
                trigger = k
        else:
            rets.append(0.0)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, trigger


print("=" * 84)
print("回撤降仓线测试（PB+EP+换手率+涨停次数 top20 剔小30%，含成本）")
print("=" * 84)
print(f"{'降仓线':<10}{'年化':>9}{'夏普':>7}{'实际最大回撤':>12}{'触发期':>8}")
print("-" * 84)

for stop in [None, 0.08, 0.12, 0.15, 0.20]:
    if stop is None:
        a, sh, dd, tg = simulate(1.0)   # 永不触发
        label = "永不降仓"
    else:
        a, sh, dd, tg = simulate(stop)
        label = f"{stop*100:.0f}%"
    tg_s = f"第{tg}期" if tg is not None else "未触发"
    print(f"{label:<10}{a:>+8.2f}%{sh:>7.2f}{dd:>+11.2f}%{tg_s:>8}")

print("-" * 84)
print("修订后目标：回撤≤15%、夏普≥0.4、跑赢沪深300")
print("结论：降仓线越浅(8%)越早清仓→收益崩；降仓线越深越接近'不降仓'。")
print("      '清仓+永久冻结'本身是问题核心：月度粒度下，降仓只能事后止血，躲不掉单月暴跌。")
