# -*- coding: utf-8 -*-
"""
① 组合增量检验：PB + 低波动 / PB + 反转
目标：验证已验证 A 股正 IC 的低波动、反转因子，能否在纯 PB 之上提供增量 alpha
（质量因子已证明"叠加≠更好"，本脚本用同一框架严格验证）

因子定义：
  PB(低好)       = -zscore(clean(PB))
  低波动(低好)   = -zscore(过去60日收益率标准差)
  反转(低好)     = -zscore(过去1月收益)   [跌得多 -> 分数高]

回测：月度调仓，top50 等权，含成本 ROUND_TRIP=0.003，区间 2015-2026
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

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
LOWVOL_WIN = 60     # 低波动窗口（交易日）
REV_WIN = 21        # 反转窗口（交易日，= 月度）


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

close = listed['close']
pb = val['pb'].reindex(close.index)
dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ 因子定义（用横截面原始值，逐换仓点算）============
def factor_lowvol(i):
    """过去60日收益率标准差（横截面）"""
    if i < LOWVOL_WIN:
        return pd.Series(np.nan, index=close.columns)
    window = close.iloc[i - LOWVOL_WIN:i]
    rets = window.pct_change().iloc[1:]
    return rets.std()


def factor_reversal(i):
    """过去1月收益（横截面）"""
    if i < REV_WIN:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[i] / close.iloc[i - REV_WIN] - 1


def score_value(i):
    return -zscore(clean(pb.iloc[i]))


FACTORS = {
    "PB(低好)": lambda i: score_value(i),
    "低波动(低好)": lambda i: -zscore(winsorize(factor_lowvol(i))),
    "反转(低好)": lambda i: -zscore(winsorize(factor_reversal(i))),
}

# ============ IC 分析 ============
print("\n" + "=" * 78)
print("各因子 IC（月度，618只，2015-2026）")
print("=" * 78)
print(f"{'因子':<16}{'IC均值':>10}{'|t|':>8}{'ICIR':>8}")
print("-" * 78)
ic_store = {}
for name, fn in FACTORS.items():
    ics = [spearman(fn(i), fwd_ret(i)) for i in rebal]
    ic_store[name] = ics
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() > 0 else 0
    print(f"{name:<16}{s.mean():>+10.4f}{tstat(ics):>8.2f}{icir:>8.2f}")

# ============ 策略回测：纯PB vs PB+低波动 vs PB+反转 ============
print("\n" + "=" * 78)
print("策略回测（top50，月度，含成本）—— 低波动/反转是否在 PB 上增量")
print("=" * 78)
print(f"{'策略':<18}{'累计':>10}{'年化':>10}{'超额':>8}{'夏普':>8}{'换手':>8}{'回撤':>8}")
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
    mdd = (cum / cum.cummax() - 1).min() * 100
    return total, annual, annual - b_annual, sharpe, np.mean(turns) * 100, mdd, cum


results = {}
pb_res = backtest(score_value)
results["纯PB"] = pb_res
print(f"{'纯PB(基准)':<18}{pb_res[0]:>+9.1f}%{pb_res[1]:>+9.2f}%"
      f"{pb_res[2]:>+7.2f}{pb_res[3]:>8.2f}{pb_res[4]:>7.1f}%{pb_res[5]:>8.2f}%")

for n in ["低波动(低好)", "反转(低好)"]:
    q_fn = FACTORS[n]
    def combo(i, q_fn=q_fn):
        return score_value(i) + q_fn(i)
    r = backtest(combo)
    results["PB+" + n] = r
    print(f"{'PB+'+n:<18}{r[0]:>+9.1f}%{r[1]:>+9.2f}%"
          f"{r[2]:>+7.2f}{r[3]:>8.2f}{r[4]:>7.1f}%{r[5]:>8.2f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

names = list(ic_store.keys())
means = [pd.Series(ic_store[n]).dropna().mean() for n in names]
axes[0].barh(names[::-1], means[::-1])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("各因子 IC（绝对值>0.03 且 |t|>2 才算有效）")
axes[0].set_xlabel("IC 均值")
axes[0].grid(alpha=0.3)

for name, r in results.items():
    axes[1].plot(r[6].values, linewidth=2, label=name)
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("纯PB vs PB+低波动 vs PB+反转 净值（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=7, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_lowvol_reversal.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_lowvol_reversal.png")
