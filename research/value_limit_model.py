# -*- coding: utf-8 -*-
"""
⑦ 涨跌停/停牌不可交易建模：PB+低波动 基准模型的"真实可成交性"检验
- 问题：回测假设调仓日能以收盘价任意买卖 top 组合，但现实中：
  1) 一字涨停(买不进)：调仓日想买入的票开盘即涨停封死，买不到
  2) 停牌(无法交易)：调仓日停牌，买不进/卖不出
  3) 一字跌停(卖不出)：调仓日想卖出的旧持仓跌停封死，卖不掉
- 建模：调仓日买入时，剔除"一字涨停+停牌"的票（顺延补入下一名）；
        卖出时，旧持仓若"一字跌停+停牌"则强制继续持有到下一期
- 对比：理想(无约束) vs 可成交(有约束) 的收益差距 = 流动性折价
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
SIZE_IND_CACHE = "size_industry_cache.pkl"
LIMIT_CACHE = "limit_cache.pkl"

HOLD, N_HOLD = 21, 100
ROUND_TRIP = 0.0030
LOWVOL_WIN = 120  # 参数微调后的最优窗口


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
with open(SIZE_IND_CACHE, 'rb') as f:
    si = pickle.load(f)
with open(LIMIT_CACHE, 'rb') as f:
    lim = pickle.load(f)

close = listed['close']
pb = val['pb'].reindex(close.index)
industry = si['industry'].reindex(close.columns)

dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]

ind_letter = industry.str.slice(0, 1).fillna('Z')
IND_DUM = {L: (ind_letter == L).astype(float) for L in ind_letter.unique() if L != 'Z'}

# 涨跌停/停牌字段（reindex 对齐到 close 的 618 列，缺失列变 NaN）
open_ = lim['open'].reindex(columns=close.columns).reindex(close.index)
high = lim['high'].reindex(columns=close.columns).reindex(close.index)
low = lim['low'].reindex(columns=close.columns).reindex(close.index)
tradestatus = lim['tradestatus'].reindex(columns=close.columns).reindex(close.index)
pct_chg = lim['pctChg'].reindex(columns=close.columns).reindex(close.index)
is_st = lim['isST'].reindex(columns=close.columns).reindex(close.index)


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def factor_lowvol(i):
    if i < LOWVOL_WIN:
        return pd.Series(np.nan, index=close.columns)
    window = close.iloc[i - LOWVOL_WIN:i]
    return window.pct_change().iloc[1:].std()


def combo_score(i):
    """行业中性化的 PB+低波动 组合得分"""
    s = -zscore(clean(pb.iloc[i])) + (-zscore(winsorize(factor_lowvol(i))))
    y = s
    X = pd.DataFrame([c for c in IND_DUM.values()]).T.assign(const=1.0)
    valid = y.notna() & X.notna().all(axis=1)
    yv = y[valid].values.astype(float)
    Xv = X[valid].values.astype(float)
    Xv = Xv - Xv.mean(axis=0)
    yv = yv - yv.mean()
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    out = pd.Series(np.nan, index=y.index)
    out[valid] = resid
    return zscore(out)


def buy_blocked(i):
    """调仓日买不进的票：一字涨停 或 停牌"""
    ts = tradestatus.iloc[i]
    halted = ts == 0
    o, h, l, c = open_.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
    # 一字涨停：open==high==low==close 且涨幅为正且接近涨停（>=5% 覆盖ST，主板10%/创业20%）
    one_word = (o == h) & (h == l) & (l == c)
    up_limit = pct_chg.iloc[i] >= 4.5  # 留容差，覆盖 ST(5%)/主板(10%)/创业(20%)
    return halted | (one_word & up_limit)


def sell_blocked(i):
    """调仓日卖不出的票：一字跌停 或 停牌"""
    ts = tradestatus.iloc[i]
    halted = ts == 0
    o, h, l, c = open_.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
    one_word = (o == h) & (h == l) & (l == c)
    down_limit = pct_chg.iloc[i] <= -4.5
    return halted | (one_word & down_limit)


def backtest(constrained):
    rets, bench, turns = [], [], []
    held = set()          # 当前持仓
    blocked_buys = 0      # 累计因涨跌停/停牌买不进被剔除的次数
    blocked_sells = 0     # 累计因跌停/停牌卖不出的次数
    prev = set()
    for i in rebal:
        sc = combo_score(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index
        bench.append(fwd[valid].mean())

        if constrained:
            # 正确逻辑：理想目标 = 按分数重排的 top N（剔除买不进的）
            # 但旧持仓里跌停/停牌的必须继续持有（占据名额），其余名额从理想目标补入
            buy_block = buy_blocked(i)
            sell_block = sell_blocked(i)

            # 理想候选：按分数排序，剔除买不进的票
            ideal = [c for c in sc[valid].nlargest(N_HOLD * 3).index
                     if c not in buy_block[buy_block].index]

            # 旧持仓里卖不出的，必须留在组合里
            must_keep = set(held) & set(sell_block[sell_block].index)
            blocked_sells += len(must_keep)

            # 先占住必须保留的，再从理想候选补满 N_HOLD
            target = set(must_keep)
            for c in ideal:
                if len(target) >= N_HOLD:
                    break
                if c not in target:
                    target.add(c)
            blocked_buys += max(0, N_HOLD - len(target))
        else:
            target = set(sc[valid].nlargest(N_HOLD).index)

        if len(target) >= 1:
            gross = fwd[list(target)].mean()
            turn = 1 - len(target & prev) / max(N_HOLD, len(target)) if prev else 1.0
            rets.append(gross - turn * ROUND_TRIP)
            turns.append(turn)
            prev = target
        held = target

    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return total, annual, annual - b_annual, sharpe, np.mean(turns) * 100, mdd, cum, blocked_buys, blocked_sells


print("=" * 78)
print("PB+低波动 可成交性检验（窗口120/持仓100/等权/行业中性）")
print("=" * 78)
print(f"{'模式':<16}{'累计':>10}{'年化':>10}{'超额':>8}{'夏普':>8}{'换手':>8}{'回撤':>8}")
print("-" * 78)

r_ideal = backtest(False)
r_real = backtest(True)

print(f"{'理想(无约束)':<16}{r_ideal[0]:>+9.1f}%{r_ideal[1]:>+9.2f}%"
      f"{r_ideal[2]:>+7.2f}{r_ideal[3]:>8.2f}{r_ideal[4]:>7.1f}%{r_ideal[5]:>8.2f}%")
print(f"{'可成交(有约束)':<16}{r_real[0]:>+9.1f}%{r_real[1]:>+9.2f}%"
      f"{r_real[2]:>+7.2f}{r_real[3]:>8.2f}{r_real[4]:>7.1f}%{r_real[5]:>8.2f}%")
print("-" * 78)
print(f"流动性折价：累计 {r_ideal[0]-r_real[0]:+.2f}pp，年化 {r_ideal[1]-r_real[1]:+.2f}pp")
print(f"约束统计：买不进剔除 {r_real[7]} 次，卖不出冻结 {r_real[8]} 次")

# ============ 图 ============
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r_ideal[6].values, linewidth=2, label='理想(无约束)')
ax.plot(r_real[6].values, linewidth=2, label='可成交(涨跌停/停牌约束)')
ax.axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
ax.set_title("PB+低波动 净值：理想 vs 可成交（含成本）")
ax.set_xlabel("换仓期")
ax.set_ylabel("累计净值")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/value_limit_model.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_limit_model.png")
