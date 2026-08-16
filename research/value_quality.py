# -*- coding: utf-8 -*-
"""
价值 + 质量双因子研究：低 PB + 高 ROE = 好公司好价格（格雷厄姆/巴菲特式）
- 价值因子：PB（低好）
- 质量因子：ROE（高好，年报滞后 4 个月避免前视偏差）
- 检验：ROE 是否在 PB 之上提供增量 alpha
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
ROE_CACHE = "roe_cache.pkl"

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def clean(f):  # 估值因子：负值置 NaN + 缩尾
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


def winsorize(s):  # ROE：保留负值，仅缩尾
    s = s.astype(float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


def tstat(ics):
    s = pd.Series(ics).dropna()
    if len(s) < 5 or s.std() == 0:
        return 0.0
    return s.mean() / s.std() * np.sqrt(len(s))


def to_bs_code(c):
    return ('sh.' if c.startswith('6') else 'sz.') + c


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(ROE_CACHE, 'rb') as f:
    roe_df = pickle.load(f)

close = listed['close']
pb = val['pb'].reindex(close.index)
roe_df.columns = [to_bs_code(c) for c in roe_df.columns]
roe_df = roe_df[[c for c in close.columns if c in roe_df.columns]]

# 点即时 ROE：年报 Y 于 Y+1 年 4/30 公布后才可用
def build_pt_roe(roe_df, dates):
    roe_pt = pd.DataFrame(index=dates, columns=roe_df.columns, dtype=float)
    for year in roe_df.index:
        start = pd.Timestamp(year + 1, 4, 30)
        end = pd.Timestamp(year + 2, 4, 30)
        mask = (dates >= start) & (dates < end)
        if mask.any():
            roe_pt.loc[mask, :] = roe_df.loc[year].values
    return roe_pt

roe_pt = build_pt_roe(roe_df, close.index)
print(f"ROE 数据: {roe_df.shape[0]}年 × {roe_df.shape[1]}只，点即时 ROE 覆盖 {roe_pt.notna().sum().sum()} 个(期×股)")

dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def score_value(i):
    return -zscore(clean(pb.iloc[i]))


def score_quality(i):
    return zscore(winsorize(roe_pt.iloc[i]))


def score_combo(i):
    return score_value(i) + score_quality(i)


# ============ IC 分析 ============
print("\n" + "=" * 78)
print("因子 IC 分析（月度，618只，2015-2026）")
print("=" * 78)
print(f"{'因子':<16}{'IC均值':>10}{'|t|':>8}{'ICIR':>8}{'方向解读':>16}")
print("-" * 78)

ic_store = {}
for name, fn in [("PB(低好)", score_value), ("ROE(高好)", score_quality),
                 ("PB+ROE", score_combo)]:
    ics = [spearman(fn(i), fwd_ret(i)) for i in rebal]
    ic_store[name] = ics
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() > 0 else 0
    interp = "低PB→高收益" if name == "PB(低好)" else ("高ROE→高收益" if name == "ROE(高好)" else "高分→高收益")
    print(f"{name:<16}{s.mean():>+10.4f}{tstat(ics):>8.2f}{icir:>8.2f}{interp:>16}")

# ============ 策略回测 ============
print("\n" + "=" * 78)
print("策略回测（top50，月度，含成本）")
print("=" * 78)
print(f"{'策略':<16}{'累计':>10}{'年化':>10}{'基准':>10}{'超额':>8}{'夏普':>8}{'换手':>8}")
print("-" * 78)

strategies = [("纯价值(PB)", score_value), ("纯质量(ROE)", score_quality),
              ("价值+质量", score_combo)]
results = {}
for name, fn in strategies:
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal:
        sc = fn(i)
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
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    results[name] = (cum, b_cum)
    print(f"{name:<16}{total:>+9.1f}%{annual:>+9.2f}%{b_annual:>+9.2f}%"
          f"{annual-b_annual:>+7.2f}{sharpe:>8.2f}{np.mean(turns)*100:>7.1f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(ic_store.keys())
means = [pd.Series(ic_store[n]).dropna().mean() for n in names]
colors = ['green' if m < 0 else 'steelblue' for m in means]
axes[0].barh(names[::-1], means[::-1], color=colors[::-1])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("因子 IC（PB 负值=低估值高收益；ROE 正值=高质量高收益）")
axes[0].set_xlabel("IC 均值")
axes[0].grid(alpha=0.3)

for name, (cum, b_cum) in results.items():
    axes[1].plot(cum.values, linewidth=2, label=name)
axes[1].plot(results['纯价值(PB)'][1].values, color='gray', linewidth=1.2,
             linestyle='--', label='等权基准')
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("价值 / 质量 / 双因子 净值对比（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_quality.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_quality.png")
