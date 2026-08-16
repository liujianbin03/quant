# -*- coding: utf-8 -*-
"""
优先级1：成本敏感性 —— 策略在真实冲击成本下还剩不剩钱？
策略 = PB+EP+换手率，top20，剔小30%，月度。
固定成本：佣金万2.5×2 + 印花税万5 = 0.001（不可压）
可变成本：滑点(冲击成本)，随股票流动性变化，是小盘股的头号杀手
ROUND_TRIP = 2×滑点 + 0.001

关键问题：滑点从 5bp 涨到 50bp，超额 alpha 何时被吃光？
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"

HOLD, N_HOLD = 21, 20
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

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]


def score_strategy(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    return pbep + t


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def backtest(round_trip):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal:
        sc = score_strategy(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        bench.append(fwd[v].mean())
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(fwd[top].mean() - turnv * round_trip)
            turns.append(turnv)
            prev = set(top)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    return annual, annual - b_annual, sharpe, np.mean(turns) * 100


SLIPS = [0.0005, 0.001, 0.002, 0.003, 0.005]   # 每边滑点 5bp/10bp/20bp/30bp/50bp

print("=" * 88)
print("成本敏感性：PB+EP+换手率 top20 剔小30%（固定成本 0.001 + 可变滑点）")
print("=" * 88)
print(f"{'每边滑点':<10}{'往返成本':>10}{'年化':>9}{'超额':>8}{'夏普':>7}{'换手':>7}")
print("-" * 88)
for slip in SLIPS:
    rt = 2 * slip + 0.001
    r = backtest(rt)
    print(f"{slip*10000:>7.0f}bp{rt*10000:>8.0f}bp{r[0]:>+8.2f}%{r[1]:>+7.2f}{r[2]:>7.2f}{r[3]:>6.1f}%")
print("-" * 88)
print("注：A股小盘股实测单边冲击成本常在 20-50bp，大盘蓝筹 5-15bp。")
print("    本策略 top20 多为银行/公用事业（大盘），但候选池含中小盘。")
