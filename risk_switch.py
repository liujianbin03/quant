# -*- coding: utf-8 -*-
"""
B：择时风险开关 —— 用市场状态决定仓位，砍掉价值策略的 -26% 回撤
在升级后策略「PB+EP + 换手率（低换手好）」之上，叠加三类宏观/市场状态开关：

  1) 趋势开关：沪深300 是否站上 200 日均线（MA200 上方=多头）
  2) 宽度开关：全市场有多少股票站上自己的 MA200（>50%=普涨，<50%=普跌）
  3) 双开关：两者同时成立才满仓

开关触发时 = 空仓（现金，收益记 0），未触发 = 持有 top20。
检验：择时是否在「几乎不损失收益」的前提下显著降低回撤、提升夏普。

口径：top20 月度；含成本；剔小30%；MA200 需 200 日暖机，故从第200日起计。
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
INDEX_NAME_CACHE = "index_name_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
MA_WIN = 200          # 均线窗口
BREADTH_THR = 0.5     # 宽度阈值


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


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)
with open(INDEX_NAME_CACHE, 'rb') as f:
    iname = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
hs300 = iname['index'].reindex(close.index)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index

# ============ 市场状态指标 ============
hs300_ma = hs300.rolling(MA_WIN).mean()
trend_on = (hs300 > hs300_ma)                       # 指数在 MA200 上方

stock_ma = close.rolling(MA_WIN, axis=0).mean()
breadth = (close > stock_ma).mean(axis=1)           # 站上MA200的股票占比
breadth_on = (breadth > BREADTH_THR)

rebal = [i for i in range(MA_WIN, len(dates), HOLD) if i + HOLD < len(dates)]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def universe_ex_small(i):
    mi = mcap.iloc[i]
    thr = mi.quantile(SMALL_PCT)
    return mi[mi > thr].index


def score_strategy(i):
    """PB+EP + 换手率（低换手好）"""
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    return pbep + t


def backtest(rule_fn):
    """rule_fn(i) -> True=满仓, False=空仓"""
    rets, bench, turns, exposure = [], [], [], []
    prev = set()
    for i in rebal:
        sc = score_strategy(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index.intersection(universe_ex_small(i))
        bench.append(fwd[valid].mean())
        on = rule_fn(i)
        exposure.append(1.0 if on else 0.0)
        if on and len(valid) >= N_HOLD:
            top = sc[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(gross - turnv * ROUND_TRIP)
            turns.append(turnv)
            prev = set(top)
        else:
            rets.append(0.0)   # 空仓=现金
            turns.append(0.0)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, annual - b_annual, sharpe, mdd, np.mean(exposure) * 100, cum


RULES = {
    "满仓(无开关)": lambda i: True,
    "趋势开关": lambda i: bool(trend_on.iloc[i]),
    "宽度开关": lambda i: bool(breadth_on.iloc[i]),
    "趋势+宽度": lambda i: bool(trend_on.iloc[i]) and bool(breadth_on.iloc[i]),
}

print("=" * 92)
print("择时风险开关：PB+EP+换手率 叠加市场状态（top20 含成本 剔小30%）")
print("=" * 92)
print(f"{'开关':<14}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}{'持仓时间':>9}")
print("-" * 92)

results = {}
for name, rule in RULES.items():
    r = backtest(rule)
    results[name] = r
    print(f"{name:<14}{r[0]:>+8.2f}%{r[1]:>+7.2f}{r[2]:>7.2f}{r[3]:>7.2f}%{r[4]:>8.0f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for name, r in results.items():
    axes[0].plot(r[5].values, linewidth=2, label=name)
axes[0].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[0].set_title("净值：不同择时开关（含成本）")
axes[0].set_xlabel("换仓期")
axes[0].set_ylabel("累计净值")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# 市场状态可视化
axes[1].plot(breadth.values, linewidth=1, alpha=0.8, label='市场宽度(站上MA200占比)')
axes[1].axhline(BREADTH_THR, color='red', linewidth=1, linestyle='--', label=f'阈值{BREADTH_THR:.0%}')
axes[1].set_title("市场宽度时序（择时依据）")
axes[1].set_xlabel("交易日")
axes[1].set_ylabel("占比")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/risk_switch.png", dpi=150)
print("\n[OK] 图表已保存 figures/risk_switch.png")
