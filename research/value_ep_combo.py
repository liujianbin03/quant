# -*- coding: utf-8 -*-
"""
PB + EP 组合检验：EP 是"第二个价值因子"，叠加到 PB 上是分散化还是稀释？
（此前已证明：质量因子叠加 PB 是稀释；EP 与 PB 同属价值，可能不同）

框架与 value_lowvol_reversal.py / ep_factor.py 完全一致：
  - 分数方向统一「越高越便宜」，等权相加（z-score 单位方差，相加≈等权排序组合）
  - 月度调仓，top50 等权，含成本 ROUND_TRIP=0.003，区间 2015-2026
  - 额外检验「剔除市值最小 30%」前后（Liu-Stambaugh-Yuan 口径）
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

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30


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


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    size = pickle.load(f)

close = listed['close']
pb = val['pb'].reindex(close.index)
pe = val['pe'].reindex(close.index)
total_share = size['totalShare'].reindex(close.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = 1.0 / pe
ep = ep.replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ 因子分数（高=便宜/好） ============
def score_pb(i):
    return -zscore(clean(pb.iloc[i]))


def score_ep(i):
    """EP 全样本（含亏损，亏损股 EP<0 自然垫底）"""
    return zscore(winsorize(ep.iloc[i]))


def score_pb_ep(i):
    """PB + EP 等权相加"""
    return score_pb(i) + score_ep(i)


FACTORS = {
    "PB(低好)":       score_pb,
    "EP全样本(高好)": score_ep,
    "PB+EP(等权)":    score_pb_ep,
}


def universe_ex_small(i, q=SMALL_PCT):
    mi = mcap.iloc[i]
    thr = mi.quantile(q)
    return mi[mi > thr].index


# ============ 因子相关性（截面平均） ============
print("=" * 90)
print("PB 与 EP 的横截面相关性（各换仓点 Spearman 取均值）")
print("=" * 90)
corrs = []
for i in rebal:
    c = spearman(score_pb(i), score_ep(i))
    if not np.isnan(c):
        corrs.append(c)
print(f"PB vs EP 截面相关 = {np.mean(corrs):+.4f}  "
      f"（越低越分散化；价值族因子通常 0.5~0.8）")

# ============ IC 分析 ============
print("\n" + "=" * 90)
print("IC 分析：PB / EP / PB+EP（全样本 vs 剔小30%）")
print("=" * 90)
print(f"{'因子':<18}{'全样本IC':>10}{'|t|':>8}{'ICIR':>8}   {'剔小30%后IC':>12}{'|t|':>8}{'ICIR':>8}")
print("-" * 90)

ic_store, ic_ex_store = {}, {}
for name, fn in FACTORS.items():
    ics_full, ics_ex = [], []
    for i in rebal:
        sc, fwd = fn(i), fwd_ret(i)
        ics_full.append(spearman(sc, fwd))
        uni = universe_ex_small(i)
        ics_ex.append(spearman(sc[uni], fwd[uni]))
    ic_store[name] = ics_full
    ic_ex_store[name] = ics_ex
    sf = pd.Series(ics_full).dropna()
    sx = pd.Series(ics_ex).dropna()
    icir_f = sf.mean() / sf.std() if sf.std() > 0 else 0
    icir_x = sx.mean() / sx.std() if sx.std() > 0 else 0
    print(f"{name:<18}{sf.mean():>+10.4f}{tstat(ics_full):>8.2f}{icir_f:>8.2f}   "
          f"{sx.mean():>+12.4f}{tstat(ics_ex):>8.2f}{icir_x:>8.2f}")
print("-" * 90)

# ============ 策略回测 ============
print("\n" + "=" * 90)
print("策略回测（top50，月度，含成本）")
print("=" * 90)


def backtest(score_fn, universe_fn=None):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal:
        sc = score_fn(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index
        if universe_fn is not None:
            valid = valid.intersection(universe_fn(i))
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
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return total, annual, annual - b_annual, sharpe, np.mean(turns) * 100, mdd, cum


print(f"{'策略':<22}{'累计':>9}{'年化':>9}{'超额':>8}{'夏普':>7}{'换手':>7}{'回撤':>8}")
print("-" * 90)

results = {}
for label, fn in [("PB(全样本)", score_pb), ("EP(全样本)", score_ep), ("PB+EP(全样本)", score_pb_ep)]:
    r = backtest(fn)
    results[label] = r
    print(f"{label:<22}{r[0]:>+8.1f}%{r[1]:>+8.2f}%{r[2]:>+7.2f}{r[3]:>7.2f}"
          f"{r[4]:>6.1f}%{r[5]:>7.2f}%")

print("-" * 90)
for label, fn in [("PB(剔小30%)", score_pb), ("EP(剔小30%)", score_ep), ("PB+EP(剔小30%)", score_pb_ep)]:
    r = backtest(fn, universe_ex_small)
    results[label] = r
    print(f"{label:<22}{r[0]:>+8.1f}%{r[1]:>+8.2f}%{r[2]:>+7.2f}{r[3]:>7.2f}"
          f"{r[4]:>6.1f}%{r[5]:>7.2f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

names = list(FACTORS.keys())
x = np.arange(len(names))
w = 0.38
means_full = [pd.Series(ic_store[n]).dropna().mean() for n in names]
means_ex = [pd.Series(ic_ex_store[n]).dropna().mean() for n in names]
axes[0].barh(x + w / 2, means_full, height=w, color='steelblue', label='全样本')
axes[0].barh(x - w / 2, means_ex, height=w, color='darkorange', label='剔小30%')
axes[0].set_yticks(x)
axes[0].set_yticklabels(names)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("IC（>0 且 |t|>2 = 有效）")
axes[0].set_xlabel("IC 均值")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, axis='x')

axes[1].plot(results["PB(全样本)"][6].values, linewidth=2, color='gray', label='PB')
axes[1].plot(results["EP(全样本)"][6].values, linewidth=2, color='steelblue', label='EP')
axes[1].plot(results["PB+EP(全样本)"][6].values, linewidth=2, color='crimson', label='PB+EP')
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("净值：PB vs EP vs PB+EP（全样本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

axes[2].plot(results["PB+EP(全样本)"][6].values, linewidth=2, color='crimson', linestyle='--', label='PB+EP·全样本')
axes[2].plot(results["PB+EP(剔小30%)"][6].values, linewidth=2, color='darkorange', label='PB+EP·剔小30%')
axes[2].plot(results["PB(剔小30%)"][6].values, linewidth=2, color='gray', linestyle='--', label='PB·剔小30%')
axes[2].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[2].set_title("净值：PB+EP 剔除小30%前后")
axes[2].set_xlabel("换仓期")
axes[2].set_ylabel("累计净值")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_ep_combo.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_ep_combo.png")
