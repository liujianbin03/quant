# -*- coding: utf-8 -*-
"""
反转 + 低换手 + 低波动 策略回测（含交易成本）
- 因子：反转(过去1月跌得多) + 低换手 + 低波动
- 持仓：top 50 等权，每月换仓
- 对比：无成本 vs 含成本 vs 等权基准
用法：python -u cost_backtest.py
"""
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

CACHE = "full_market_cache.pkl"
LOOKBACK = 126   # 6个月（波动窗口）
REV = 21         # 1个月（反转/换手窗口）
HOLD = 21        # 持有1个月
N_HOLD = 50      # 持仓数量

# A股真实交易成本
COMMISSION = 0.00025   # 佣金 万2.5（双边）
STAMP_TAX = 0.0005     # 印花税 万5（仅卖出）
SLIPPAGE = 0.001       # 滑点 千1（双边）
BUY_COST = COMMISSION + SLIPPAGE              # 0.125%
SELL_COST = COMMISSION + STAMP_TAX + SLIPPAGE  # 0.175%
ROUND_TRIP = BUY_COST + SELL_COST             # 0.30%


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


# ============ 数据 ============
with open(CACHE, 'rb') as f:
    data = pickle.load(f)
close_df = data['close']
turn_df = data['turn']
daily_ret = close_df.pct_change(fill_method=None)
dates = close_df.index
print(f"数据：{close_df.shape[0]} 交易日 × {close_df.shape[1]} 只股票")

# ============ 回测 ============
i = LOOKBACK
prev_held = set()
rets_no, rets_cost, rets_bench = [], [], []
turnovers = []

while i + HOLD < len(dates):
    t0 = dates[i - LOOKBACK]
    t1 = dates[i]
    t_rev = dates[i - REV]
    t2 = dates[i + HOLD]

    rev = close_df.loc[t1] / close_df.loc[t_rev] - 1   # 过去1月收益（反转：越低越好）
    vol = daily_ret.loc[t0:t1].std()                    # 过去6月波动率（越低越好）
    turn = turn_df.loc[t_rev:t1].mean()                 # 过去1月换手率（越低越好）

    valid = rev.dropna().index.intersection(vol.dropna().index)\
                            .intersection(turn.dropna().index)
    if len(valid) < N_HOLD:
        i += HOLD
        continue

    score = -zscore(rev[valid]) - zscore(vol[valid]) - zscore(turn[valid])
    top = score.nlargest(N_HOLD).index

    fwd = close_df.loc[t2] / close_df.loc[t1] - 1
    gross = fwd[top].mean()          # 组合毛收益
    bench = fwd[valid].mean()        # 等权全市场基准

    # 换手率：本期持仓与上期的差异比例
    if prev_held:
        overlap = len(set(top) & prev_held)
        turnover = 1 - overlap / N_HOLD
    else:
        turnover = 1.0
    cost = turnover * ROUND_TRIP   # 本期换仓成本

    rets_no.append(gross)
    rets_cost.append(gross - cost)
    rets_bench.append(bench)
    turnovers.append(turnover)

    prev_held = set(top)
    i += HOLD

# ============ 统计 ============
def stats(rets):
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    return total, annual, dd, sharpe, cum

t_no, a_no, d_no, sh_no, cum_no = stats(rets_no)
t_cost, a_cost, d_cost, sh_cost, cum_cost = stats(rets_cost)
t_b, a_b, d_b, sh_b, cum_b = stats(rets_bench)

avg_turn = np.mean(turnovers) * 100
n_periods = len(rets_no)

print("=" * 78)
print(f"反转+低换手+低波动 策略回测（top{N_HOLD}等权，{n_periods}期，2015-2026）")
print(f"单边成本：买{ BUY_COST*100:.3f}% / 卖{ SELL_COST*100:.3f}%，双边 {ROUND_TRIP*100:.2f}%")
print(f"平均换手率（每期）：{avg_turn:.1f}%")
print("=" * 78)
print(f"{'组合':<22}{'累计':>9}{'年化':>9}{'回撤':>9}{'夏普':>7}")
print("-" * 78)
print(f"{'反转策略(无成本)':<20}{t_no:>+9.2f}%{a_no:>+9.2f}%{d_no:>9.2f}%{sh_no:>7.2f}")
print(f"{'反转策略(含成本)':<20}{t_cost:>+9.2f}%{a_cost:>+9.2f}%{d_cost:>9.2f}%{sh_cost:>7.2f}")
print(f"{'等权全市场基准':<20}{t_b:>+9.2f}%{a_b:>+9.2f}%{d_b:>9.2f}%{sh_b:>7.2f}")
print("-" * 78)
print(f"成本侵蚀：累计收益被吃掉 {t_no - t_cost:.2f} 个百分点")
print(f"          年化收益被吃掉 {a_no - a_cost:.2f} 个百分点")
print(f"          夏普从 {sh_no:.2f} 降到 {sh_cost:.2f}")

# ============ 图 ============
plt.figure(figsize=(14, 8))
plt.plot(cum_no.index, cum_no.values, label="反转策略(无成本)", color='red', linewidth=2.2)
plt.plot(cum_cost.index, cum_cost.values, label="反转策略(含成本)", color='orange', linewidth=2.2)
plt.plot(cum_b.index, cum_b.values, label="等权全市场基准", color='gray', linewidth=1.8, linestyle='--')
plt.axhline(1.0, color='black', linestyle=':', alpha=0.5)
plt.title(f"交易成本对反转策略的影响（平均换手 {avg_turn:.0f}%/期，双边成本 {ROUND_TRIP*100:.2f}%）")
plt.xlabel("换仓期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cost_backtest.png", dpi=150)
print("\n[OK] 图表已保存 cost_backtest.png")
