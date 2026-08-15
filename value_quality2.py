# -*- coding: utf-8 -*-
"""
价值 + 质量（多质量因子版）：检验哪种"质量"能在 PB 之上提供增量 alpha
候选质量因子：
  单期类（quality_cache.pkl，4个月年报滞后）：
    毛利率(高好) / 净利增速(高好) / 营收增速(高好) / 负债率(低好)
  稳定性类（roe_cache.pkl，5年窗口）：
    ROE均值(高好) / ROE波动率(低好=稳定)
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
QUALITY_CACHE = "quality_cache.pkl"

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
WIN = 5  # 稳定性窗口（年）


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
dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]

roe_df.columns = [to_bs_code(c) for c in roe_df.columns]
roe_df = roe_df[[c for c in close.columns if c in roe_df.columns]]


def build_pt(annual_df, dates):
    """年度 DataFrame(index=year) -> 点即时 DataFrame(index=date)，4个月滞后"""
    pt = pd.DataFrame(index=dates, columns=annual_df.columns, dtype=float)
    for year in annual_df.index:
        start = pd.Timestamp(year + 1, 4, 30)
        end = pd.Timestamp(year + 2, 4, 30)
        mask = (dates >= start) & (dates < end)
        if mask.any():
            pt.loc[mask, :] = annual_df.loc[year].values
    return pt


# 稳定性/均值因子：每个换仓点取"已公布"的近 5 年年报 ROE
def build_roe_stats(roe_df, dates, rebal_idx, win=WIN):
    """返回 (roe_mean_pt, roe_std_pt) 两个 DataFrame，只填换仓点"""
    cols = roe_df.columns
    mean_pt = pd.DataFrame(index=dates, columns=cols, dtype=float)
    std_pt = pd.DataFrame(index=dates, columns=cols, dtype=float)
    years = list(roe_df.index)
    for i in rebal_idx:
        t = dates[i]
        avail = [y for y in years if pd.Timestamp(y + 1, 4, 30) <= t]
        if len(avail) < 2:
            continue
        win_years = avail[-win:]
        sub = roe_df.loc[win_years]
        mean_pt.iloc[i] = sub.mean().values
        std_pt.iloc[i] = sub.std().values
    return mean_pt, std_pt


# 单期质量因子（quality_cache）
try:
    with open(QUALITY_CACHE, 'rb') as f:
        qc = pickle.load(f)
    q_pt = {}
    for name, df in qc.items():
        df.columns = [to_bs_code(c) for c in df.columns]
        df = df[[c for c in close.columns if c in df.columns]]
        q_pt[name] = build_pt(df, dates)
    print("[OK] 已加载 quality_cache.pkl")
except FileNotFoundError:
    q_pt = {}
    print("[警告] quality_cache.pkl 不存在，仅测 ROE 稳定性类因子")

roe_mean_pt, roe_std_pt = build_roe_stats(roe_df, dates, rebal)


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ 因子定义 ============
def score_value(i):
    return -zscore(clean(pb.iloc[i]))


FACTORS = {
    "PB(低好)": lambda i: score_value(i),
    "ROE单年(高好)": lambda i: zscore(winsorize(build_pt(roe_df, dates).iloc[i])),
    "ROE均值5y(高好)": lambda i: zscore(winsorize(roe_mean_pt.iloc[i])),
    "ROE波动5y(低好)": lambda i: -zscore(winsorize(roe_std_pt.iloc[i])),
}
if q_pt:
    FACTORS.update({
        "毛利率(高好)": lambda i, n='毛利率': zscore(winsorize(q_pt[n].iloc[i])),
        "净利增速(高好)": lambda i, n='净利增速': zscore(winsorize(q_pt[n].iloc[i], 0.02)),
        "营收增速(高好)": lambda i, n='营收增速': zscore(winsorize(q_pt[n].iloc[i], 0.02)),
        "负债率(低好)": lambda i, n='负债率': -zscore(winsorize(q_pt[n].iloc[i])),
    })

# ============ IC 分析 ============
print("\n" + "=" * 78)
print("各质量因子 IC（月度，618只，2015-2026）")
print("=" * 78)
print(f"{'因子':<18}{'IC均值':>10}{'|t|':>8}{'ICIR':>8}")
print("-" * 78)
ic_store = {}
for name, fn in FACTORS.items():
    ics = [spearman(fn(i), fwd_ret(i)) for i in rebal]
    ic_store[name] = ics
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() > 0 else 0
    print(f"{name:<18}{s.mean():>+10.4f}{tstat(ics):>8.2f}{icir:>8.2f}")

# ============ 策略回测：纯PB vs PB+各质量因子 ============
print("\n" + "=" * 78)
print("策略回测（top50，月度，含成本）—— 质量因子是否在 PB 上增量")
print("=" * 78)
print(f"{'策略':<20}{'累计':>10}{'年化':>10}{'超额':>8}{'夏普':>8}{'换手':>8}")
print("-" * 78)


def backtest(fn):
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
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    return total, annual, annual - b_annual, sharpe, np.mean(turns) * 100, cum


results = {}
pb_res = backtest(score_value)
results["纯PB"] = pb_res
print(f"{'纯PB(基准)':<20}{pb_res[0]:>+9.1f}%{pb_res[1]:>+9.2f}%"
      f"{pb_res[2]:>+7.2f}{pb_res[3]:>8.2f}{pb_res[4]:>7.1f}%")

combo_names = [n for n in FACTORS if n != "PB(低好)"]
for n in combo_names:
    q_fn = FACTORS[n]
    def combo(i, q_fn=q_fn):
        return score_value(i) + q_fn(i)
    r = backtest(combo)
    results["PB+" + n] = r
    print(f"{'PB+'+n:<20}{r[0]:>+9.1f}%{r[1]:>+9.2f}%"
          f"{r[2]:>+7.2f}{r[3]:>8.2f}{r[4]:>7.1f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

names = list(ic_store.keys())
means = [pd.Series(ic_store[n]).dropna().mean() for n in names]
axes[0].barh(names[::-1], means[::-1])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("各质量因子 IC（绝对值>0.03 且 |t|>2 才算有效）")
axes[0].set_xlabel("IC 均值")
axes[0].grid(alpha=0.3)

for name, r in results.items():
    axes[1].plot(r[5].values, linewidth=2, label=name)
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("纯PB vs PB+各质量因子 净值（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=7, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_quality2.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_quality2.png")
