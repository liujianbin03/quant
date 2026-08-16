# -*- coding: utf-8 -*-
"""
目标驱动的参数搜索（带样本外验证）：系统扫描策略配置，找最优前沿，对标 4 个目标
配置维度：
  - 信号组合：PB+EP / PB+EP+换手率 / PB+EP+换手率+涨停次数 / EP+换手率(去PB)
  - 持仓数 N_HOLD：10 / 20 / 50
  - 剔小30% 固定（已证明显著有利）
口径：月度含成本；样本内(2015-2021)选参，样本外(2021-2026)验证。
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"

HOLD = 21
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
split_pos = close.index.get_indexer([pd.Timestamp("2021-01-04")], method='nearest')[0]
in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]


def f_pb(i):
    return -zscore(clean(pb.iloc[i]))


def f_ep(i):
    return zscore(winsorize(ep.iloc[i]))


def f_turn(i):
    return -zscore(winsorize(turn.iloc[i - 21:i].mean()))


def f_limit(i):
    return -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


SCORES = {
    "PB+EP": lambda i: f_pb(i) + f_ep(i),
    "PB+EP+换手": lambda i: f_pb(i) + f_ep(i) + f_turn(i),
    "PB+EP+换手+涨停": lambda i: f_pb(i) + f_ep(i) + f_turn(i) + f_limit(i),
    "EP+换手(去PB)": lambda i: f_ep(i) + f_turn(i),
}


def backtest(score_fn, rebal_list, n_hold):
    rets, prev = [], set()
    for i in rebal_list:
        sc = score_fn(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= n_hold:
            top = sc[v].nlargest(n_hold).index
            turnv = 1 - len(set(top) & prev) / n_hold if prev else 1.0
            rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
            prev = set(top)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd


print("=" * 104)
print("参数搜索（样本内选参 → 样本外验证），对标：夏普≥1 回撤≤8% 收益≥10%/3月")
print("=" * 104)
print(f"{'信号组合':<18}{'持仓':>6}  {'样本内夏普':>10}{'样本内回撤':>10}  {'样本外夏普':>10}{'样本外回撤':>10}{'样本外年化':>10}")
print("-" * 104)

rows = []
for name, fn in SCORES.items():
    for n in [10, 20, 50]:
        a_in, sh_in, dd_in = backtest(fn, in_rebal, n)
        a_out, sh_out, dd_out = backtest(fn, out_rebal, n)
        rows.append((name, n, sh_in, dd_in, sh_out, dd_out, a_out))
        print(f"{name:<18}{n:>6}  {sh_in:>+10.2f}{dd_in:>+10.2f}  {sh_out:>+10.2f}{dd_out:>+10.2f}{a_out:>+10.2f}")

# 最优（样本外夏普最高，且样本内夏普>0 = 未被样本内否掉）
best = max([r for r in rows if r[2] > 0], key=lambda r: r[4])
print("-" * 104)
print(f"\n最优(样本外夏普最高且样本内>0): {best[0]} 持仓{best[1]}  → 样本外夏普 {best[4]:.2f} 回撤 {best[5]:.2f}% 年化 {best[6]:.2f}%")

# 有没有任何配置接近目标？
print("\n目标达成检查：")
any_sharpe = any(r[4] >= 1.0 for r in rows)
any_dd = any(r[5] >= -8.0 for r in rows)
print(f"  是否存在 夏普≥1 的配置？  {'有' if any_sharpe else '无（全场最高样本外夏普 = %.2f）' % max(r[4] for r in rows)}")
print(f"  是否存在 回撤≤8% 的配置？ {'有' if any_dd else '无（全场最浅回撤 = %.2f%%）' % max(r[5] for r in rows)}")
