# -*- coding: utf-8 -*-
"""
价值因子样本外验证：2021-01-04 切分（月度换仓，步进 21 天）
- 样本内（2015-2021）vs 样本外（2021-2026）的 IC 方向 + 策略表现
- 检验：价值溢价是真实规律，还是 2015 牛市过拟合？
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
SPLIT = "2021-01-04"

HOLD = 21
N_HOLD = 50
ROUND_TRIP = 0.0030


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


def clean(f):
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


def tstat(ics):
    s = pd.Series(ics).dropna()
    if len(s) < 5 or s.std() == 0:
        return 0.0
    return s.mean() / s.std() * np.sqrt(len(s))


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)

close = listed['close']
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
ps = val['ps'].reindex(close.index)
pcf = val['pcf'].reindex(close.index)
FACTORS = {"PE": pe, "PB": pb, "PS": ps, "PCF": pcf}

dates = close.index
split_pos = close.index.get_indexer([pd.Timestamp(SPLIT)], method='nearest')[0]
print(f"切分点: {dates[split_pos]}（第 {split_pos} 天 / 共 {len(dates)} 天）")

# 月度换仓日期（步进 21 天），再切成样本内/外
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]
print(f"月度换仓期：样本内 {len(in_rebal)} 期，样本外 {len(out_rebal)} 期")


def build_value_score(t):
    return -sum(zscore(clean(f.iloc[t])) for f in FACTORS.values())


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ IC 分析（分期间） ============
print("\n" + "=" * 78)
print("价值因子 IC 分期间：样本内 vs 样本外（月度）")
print("=" * 78)
print(f"{'因子':<10}{'样本内IC':>10}{'样本内|t|':>10}{'样本外IC':>10}{'样本外|t|':>10}{'方向':>12}")
print("-" * 78)
ic_in_all, ic_out_all = {}, {}
for name, fdf in FACTORS.items():
    ics_in = [spearman(clean(fdf.iloc[i]), fwd_ret(i)) for i in in_rebal]
    ics_out = [spearman(clean(fdf.iloc[i]), fwd_ret(i)) for i in out_rebal]
    ic_in_all[name] = pd.Series(ics_in).dropna().mean()
    ic_out_all[name] = pd.Series(ics_out).dropna().mean()
    m_in = ic_in_all[name]
    m_out = ic_out_all[name]
    dir_in = "低→高" if m_in < 0 else "低→低"
    dir_out = "低→高" if m_out < 0 else "低→低"
    print(f"{name:<10}{m_in:>+10.4f}{tstat(ics_in):>10.2f}{m_out:>+10.4f}{tstat(ics_out):>10.2f}{dir_in+'/'+dir_out:>12}")

com_in = [spearman(build_value_score(i), fwd_ret(i)) for i in in_rebal]
com_out = [spearman(build_value_score(i), fwd_ret(i)) for i in out_rebal]
m_in, m_out = pd.Series(com_in).dropna().mean(), pd.Series(com_out).dropna().mean()
print(f"{'合成':<10}{m_in:>+10.4f}{tstat(com_in):>10.2f}{m_out:>+10.4f}{tstat(com_out):>10.2f}{'正/正':>12}")

# ============ 策略分期间回测 ============
print("\n" + "=" * 78)
print("低估值策略分期间（top50，月度，含成本）")
print("=" * 78)


def backtest(rebal_list):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal_list:
        sc = build_value_score(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index
        bench.append(fwd[valid].mean())
        if len(valid) >= N_HOLD:
            top = sc[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            turn = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(gross - turn * ROUND_TRIP)
            turns.append(turn)
            prev = set(top)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    return annual, b_annual, dd, sharpe, np.mean(turns) * 100, cum, b_cum


a_in, b_in, d_in, s_in, t_in, cum_in, bcum_in = backtest(in_rebal)
a_out, b_out, d_out, s_out, t_out, cum_out, bcum_out = backtest(out_rebal)

print(f"{'期间':<10}{'策略年化':>12}{'基准年化':>12}{'超额':>10}{'回撤':>10}{'夏普':>8}{'换手':>8}")
print("-" * 78)
print(f"{'样本内':<10}{a_in:>+12.2f}{b_in:>+12.2f}{a_in-b_in:>+10.2f}{d_in:>10.2f}{s_in:>8.2f}{t_in:>8.1f}")
print(f"{'样本外':<10}{a_out:>+12.2f}{b_out:>+12.2f}{a_out-b_out:>+10.2f}{d_out:>10.2f}{s_out:>8.2f}{t_out:>8.1f}")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

full_cum = pd.concat([cum_in, cum_out]).reset_index(drop=True)
full_b = pd.concat([bcum_in, bcum_out]).reset_index(drop=True)
axes[0].plot(full_cum.values, color='green', linewidth=2, label='低估值策略(含成本)')
axes[0].plot(full_b.values, color='gray', linewidth=1.5, linestyle='--', label='等权基准')
axes[0].axvline(len(cum_in), color='red', linewidth=1, linestyle=':', label='样本外分界(2021)')
axes[0].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[0].set_title("低估值策略净值（含成本）")
axes[0].set_xlabel("换仓期")
axes[0].set_ylabel("累计净值")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

names = list(FACTORS.keys()) + ["合成"]
ins = [ic_in_all[n] for n in FACTORS] + [m_in]
outs = [ic_out_all[n] for n in FACTORS] + [m_out]
x = np.arange(len(names))
w = 0.35
axes[1].bar(x - w/2, ins, w, label='样本内', color='steelblue')
axes[1].bar(x + w/2, outs, w, label='样本外', color='orange')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_title("价值因子 IC：样本内 vs 样本外")
axes[1].set_ylabel("IC 均值（负=低估值高收益）")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_oos.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_oos.png")
