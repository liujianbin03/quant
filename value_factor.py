# -*- coding: utf-8 -*-
"""
价值因子研究：PE/PB/PS/PCF 的 IC 分析 + 低估值策略回测
- 假设：低估值（低PE/低PB/低PS/低PCF）→ 未来收益更高
- 优势：换手低、成本低、退市风险小（对比反转策略）
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

HOLD = 21  # 月度换仓
N_HOLD = 50
COMMISSION = 0.00025
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
ROUND_TRIP = COMMISSION + SLIPPAGE + COMMISSION + STAMP_TAX + SLIPPAGE  # 0.30%


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
    """估值因子清洗：负值(亏损/负资产)置 NaN，然后 1%~99% 缩尾"""
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
pe_raw, pb_raw = val['pe'], val['pb']
ps_raw, pcf_raw = val['ps'], val['pcf']

# 对齐日期
pe, pb = pe_raw.reindex(close.index), pb_raw.reindex(close.index)
ps, pcf = ps_raw.reindex(close.index), pcf_raw.reindex(close.index)

dates = close.index
daily_ret = close.pct_change(fill_method=None)

FACTORS = {"PE": pe, "PB": pb, "PS": ps, "PCF": pcf}

# ============ IC 分析 ============
print("=" * 78)
print("价值因子 IC 分析（月度，2015-2026，618只）")
print("=" * 78)
print(f"{'因子':<10}{'IC均值':>10}{'|t|':>8}{'ICIR':>8}{'方向':>14}")
print("-" * 78)
ic_store = {}
for name, fdf in FACTORS.items():
    ics = []
    i = 0
    while i + HOLD < len(dates):
        f = clean(fdf.iloc[i])
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        ic = spearman(f, fwd)
        if not np.isnan(ic):
            ics.append(ic)
        i += HOLD
    ic_store[name] = ics
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() > 0 else 0
    direction = "低估值→高收益" if s.mean() < 0 else "低估值→低收益"
    print(f"{name:<10}{s.mean():>+10.4f}{tstat(ics):>8.2f}{icir:>8.2f}{direction:>14}")
print("-" * 78)
print("注：IC<0 且 |t|>2 说明「低估值→高收益」成立（价值因子有效）")

# ============ 合成价值因子 ============
# 每个因子「越低越好」，合成分 = -(zPE+zPB+zPS+zPCF)，越高越便宜
def build_value_score(t):
    parts = []
    for name, fdf in FACTORS.items():
        parts.append(zscore(clean(fdf.iloc[t])))
    return -sum(parts)

composite_ics = []
i = 0
while i + HOLD < len(dates):
    sc = build_value_score(i)
    fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
    ic = spearman(sc, fwd)
    if not np.isnan(ic):
        composite_ics.append(ic)
    i += HOLD
cs = pd.Series(composite_ics).dropna()
print(f"\n合成价值因子 IC = {cs.mean():+.4f}，|t| = {tstat(composite_ics):.2f}，ICIR = {cs.mean()/cs.std():.2f}")

# ============ 价值策略回测 ============
print("\n" + "=" * 78)
print("低估值策略回测（合成价值分 top50，月度换仓，含成本）")
print("=" * 78)

for label, N in [("top50", 50)]:
    rets, bench_rets, turnovers = [], [], []
    prev_held = set()
    i = 0
    while i + HOLD < len(dates):
        sc = build_value_score(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        valid = sc.dropna().index
        bench_rets.append(fwd[valid].mean())
        if len(valid) >= N:
            top = sc[valid].nlargest(N).index
            gross = fwd[top].mean()
            turnover = 1 - len(set(top) & prev_held) / N if prev_held else 1.0
            rets.append(gross - turnover * ROUND_TRIP)
            turnovers.append(turnover)
            prev_held = set(top)
        i += HOLD

    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    avg_turn = np.mean(turnovers) * 100

    b = pd.Series(bench_rets)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100

    print(f"{label}: 累计 {total:+.2f}%  年化 {annual:+.2f}%  回撤 {dd:.2f}%  夏普 {sharpe:.2f}")
    print(f"       平均换手率 {avg_turn:.1f}%/期   （反转策略 55.2%/期）")
    print(f"       等权基准 年化 {b_annual:+.2f}%  超额 {annual - b_annual:+.2f} 个百分点")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IC 柱状图
names = list(FACTORS.keys()) + ["合成"]
means = [pd.Series(ic_store[n]).dropna().mean() for n in FACTORS] + [cs.mean()]
colors = ['green' if m < 0 else 'red' for m in means]
axes[0].barh(names[::-1], means[::-1], color=colors[::-1])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("价值因子 IC（负值=低估值高收益）")
axes[0].set_xlabel("IC 均值")
axes[0].grid(alpha=0.3)

# 净值曲线
s = pd.Series(rets)
cum = (1 + s).cumprod()
b = pd.Series(bench_rets)
b_cum = (1 + b).cumprod()
axes[1].plot(cum.values, color='green', linewidth=2, label='低估值策略(含成本)')
axes[1].plot(b_cum.values, color='gray', linewidth=1.5, linestyle='--', label='等权基准')
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("低估值策略 vs 等权基准")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("value_factor.png", dpi=150)
print("\n[OK] 图表已保存 value_factor.png")
