# -*- coding: utf-8 -*-
"""
样本外验证：数据切两段
- 样本内 2015-2020：挖因子、看 IC（"发现"阶段）
- 样本外 2021-2026：同样的因子，看还灵不灵（"验证"阶段）
策略：反转(1月) + 低换手 + 低波动，top50 等权，含成本
用法：python -u oos_validation.py
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
LOOKBACK = 126   # 6个月波动窗口
REV = 21         # 1个月反转窗口
HOLD = 21        # 持有1个月
N_HOLD = 50      # 持仓数量

COMMISSION = 0.00025
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
BUY_COST = COMMISSION + SLIPPAGE
SELL_COST = COMMISSION + STAMP_TAX + SLIPPAGE
ROUND_TRIP = BUY_COST + SELL_COST

SPLIT = "2021-01-01"

FACTORS = {
    "动量(6月)": +1,
    "反转(1月)": -1,
    "波动率(6月)": -1,
    "换手率(1月)": -1,
}


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


# ============ 数据 ============
with open(CACHE, 'rb') as f:
    data = pickle.load(f)
close_df = data['close']
turn_df = data['turn']
daily_ret = close_df.pct_change(fill_method=None)
dates = close_df.index

split_pos = int((dates >= pd.Timestamp(SPLIT)).argmax())
print(f"数据：{close_df.shape[0]} 交易日 × {close_df.shape[1]} 只")
print(f"切分点：{dates[split_pos].date()}  |  样本内 {dates[LOOKBACK].date()}~{dates[split_pos-1].date()}"
      f"  |  样本外 {dates[split_pos].date()}~{dates[-1].date()}")


def run_period(start_i, end_i):
    """在 [start_i, end_i) 期间：收集各因子 IC + 跑策略(含成本) + 等权基准"""
    ic_collect = {n: [] for n in FACTORS}
    rets, bench_rets, turnovers = [], [], []
    prev_held = set()
    i = start_i
    while i + HOLD < end_i:
        t0 = dates[i - LOOKBACK]
        t1 = dates[i]
        t_rev = dates[i - REV]
        t2 = dates[i + HOLD]

        mom = close_df.loc[t1] / close_df.loc[t0] - 1
        rev = close_df.loc[t1] / close_df.loc[t_rev] - 1
        vol = daily_ret.loc[t0:t1].std()
        turn = turn_df.loc[t_rev:t1].mean()
        fwd = close_df.loc[t2] / close_df.loc[t1] - 1

        fmap = {"动量(6月)": mom, "反转(1月)": rev,
                "波动率(6月)": vol, "换手率(1月)": turn}
        for n in FACTORS:
            ic = spearman(fmap[n], fwd)
            if not np.isnan(ic):
                ic_collect[n].append(ic)

        valid = rev.dropna().index.intersection(vol.dropna().index)\
                            .intersection(turn.dropna().index)
        bench_rets.append(fwd[valid].mean())

        if len(valid) >= N_HOLD:
            score = -zscore(rev[valid]) - zscore(vol[valid]) - zscore(turn[valid])
            top = score.nlargest(N_HOLD).index
            gross = fwd[top].mean()
            if prev_held:
                turnover = 1 - len(set(top) & prev_held) / N_HOLD
            else:
                turnover = 1.0
            rets.append(gross - turnover * ROUND_TRIP)
            turnovers.append(turnover)
            prev_held = set(top)

        i += HOLD
    return ic_collect, pd.Series(rets), pd.Series(bench_rets), turnovers


def stats(rets):
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    return total, annual, dd, sharpe, cum


# ============ 跑两段 ============
in_ic, in_rets, in_bench, in_turn = run_period(LOOKBACK, split_pos)
out_ic, out_rets, out_bench, out_turn = run_period(split_pos, len(dates))

print("\n" + "=" * 84)
print("因子 IC 对比（样本内 vs 样本外）—— 看因子方向是否稳定")
print("=" * 84)
print(f"{'因子':<14}{'样本内IC':>10}{'样本内|t|':>11}{'样本外IC':>10}{'样本外|t|':>11}{'方向稳定?':>10}")
print("-" * 84)
for n, direction in FACTORS.items():
    s_in = pd.Series(in_ic[n]).dropna()
    s_out = pd.Series(out_ic[n]).dropna()
    ic_in = s_in.mean()
    ic_out = s_out.mean()
    t_in = abs(ic_in / s_in.std()) * np.sqrt(len(s_in)) if s_in.std() > 0 else 0
    t_out = abs(ic_out / s_out.std()) * np.sqrt(len(s_out)) if s_out.std() > 0 else 0
    stable = "是" if np.sign(ic_in) == np.sign(ic_out) and ic_in != 0 else "否(翻车)"
    print(f"{n:<14}{ic_in:>10.4f}{t_in:>11.2f}{ic_out:>10.4f}{t_out:>11.2f}{stable:>10}")
print("-" * 84)
print("注：样本外若 IC 方向翻转或 |t|<2，说明因子在样本内是过拟合/失效")

# ============ 策略表现对比 ============
ti, ai, di, si, cum_in = stats(in_rets)
to_, ao, do_, so, cum_out = stats(out_rets)
bi, abi, dbi, sbi, _ = stats(in_bench)
bo, abo, dbo, sbo, _ = stats(out_bench)

print("\n" + "=" * 84)
print("策略表现对比（反转+低换手+低波动，top50，含成本）")
print("=" * 84)
print(f"{'区间':<22}{'累计':>9}{'年化':>9}{'回撤':>9}{'夏普':>7}")
print("-" * 84)
print(f"{'样本内 2015-2020':<20}{ti:>+9.2f}%{ai:>+9.2f}%{di:>9.2f}%{si:>7.2f}")
print(f"{'样本外 2021-2026':<20}{to_:>+9.2f}%{ao:>+9.2f}%{do_:>9.2f}%{so:>7.2f}")
print(f"{'样本内 等权基准':<20}{bi:>+9.2f}%{abi:>+9.2f}%{dbi:>9.2f}%{sbi:>7.2f}")
print(f"{'样本外 等权基准':<20}{bo:>+9.2f}%{abo:>+9.2f}%{dbo:>9.2f}%{sbo:>7.2f}")
print("-" * 84)
print(f"平均换手率：样本内 {np.mean(in_turn)*100:.1f}%  样本外 {np.mean(out_turn)*100:.1f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左：全期净值（含成本）+ 切分线
full_rets = pd.concat([in_rets, out_rets])
full_bench = pd.concat([in_bench, out_bench])
cum_full = (1 + full_rets).cumprod()
cum_fullb = (1 + full_bench).cumprod()
axes[0].plot(cum_full.index, cum_full.values, color='red', linewidth=2, label='反转策略(含成本)')
axes[0].plot(cum_fullb.index, cum_fullb.values, color='gray', linewidth=1.5, linestyle='--', label='等权基准')
axes[0].axvline(len(in_rets), color='blue', linestyle='--', alpha=0.7, label='样本内/外分界(2021)')
axes[0].axhline(1.0, color='black', linestyle=':', alpha=0.4)
axes[0].set_title("全期净值（含成本）")
axes[0].set_xlabel("换仓期")
axes[0].set_ylabel("累计净值")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右：两段年化收益对比柱状
labels = ['样本内\n策略', '样本内\n基准', '样本外\n策略', '样本外\n基准']
vals = [ai, abi, ao, abo]
colors = ['red', 'gray', 'red', 'gray']
bars = axes[1].bar(labels, vals, color=colors, alpha=0.8)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width()/2, v + (0.2 if v >= 0 else -0.6),
                 f"{v:.1f}%", ha='center', fontsize=10)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title("年化收益对比：样本内 vs 样本外")
axes[1].set_ylabel("年化收益(%)")
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figures/oos_validation.png", dpi=150)
print("\n[OK] 图表已保存 figures/oos_validation.png")
