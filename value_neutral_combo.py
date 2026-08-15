# -*- coding: utf-8 -*-
"""
③ PB+低波动 组合的双重中性化：剥离市值/行业暴露后，纯 alpha 还剩多少

组合得分 combo = PB(低好) + 低波动(低好)
中性化 = 对 combo 得分做横截面 OLS，回归 log(市值) / 行业哑变量 取残差

对比：原始 combo / 市值中性 / 行业中性 / 双重中性
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

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
LOWVOL_WIN = 60


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
with open(SIZE_IND_CACHE, 'rb') as f:
    si = pickle.load(f)

close = listed['close']
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
industry = si['industry'].reindex(close.columns)

dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]

mcap = close.mul(total_share, axis=1)
log_mcap = np.log(mcap.where(mcap > 0))
ind_letter = industry.str.slice(0, 1).fillna('Z')


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def factor_lowvol(i):
    if i < LOWVOL_WIN:
        return pd.Series(np.nan, index=close.columns)
    window = close.iloc[i - LOWVOL_WIN:i]
    rets = window.pct_change().iloc[1:]
    return rets.std()


def combo_score(i):
    """PB(低好) + 低波动(低好)"""
    return -zscore(clean(pb.iloc[i])) + (-zscore(winsorize(factor_lowvol(i))))


def neutral_resid(i, y_series, controls):
    """对 y_series 做横截面 OLS，返回残差（已 zscore）"""
    y = y_series
    X = []
    for c in controls:
        if isinstance(c, pd.DataFrame):
            x = c.iloc[i].astype(float)
        else:
            x = c.astype(float)
        X.append(x)
    X = pd.DataFrame(X).T.assign(const=1.0)
    valid = y.notna() & X.notna().all(axis=1)
    yv = y[valid].values.astype(float)
    Xv = X[valid].values.astype(float)
    if len(yv) < 30:
        return pd.Series(np.nan, index=y.index)
    Xv = Xv - Xv.mean(axis=0)
    yv = yv - yv.mean()
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    out = pd.Series(np.nan, index=y.index)
    out[valid] = resid
    return zscore(out)


IND_DUM = {L: (ind_letter == L).astype(float) for L in ind_letter.unique() if L != 'Z'}

FACTORS = {
    "PB+低波动(原始)": lambda i: combo_score(i),
    "PB+低波动(市值中性)": lambda i: neutral_resid(i, combo_score(i), [log_mcap]),
    "PB+低波动(行业中性)": lambda i: neutral_resid(i, combo_score(i), list(IND_DUM.values())),
    "PB+低波动(双重中性)": lambda i: neutral_resid(i, combo_score(i), [log_mcap] + list(IND_DUM.values())),
}

# ============ IC 分析 ============
print("\n" + "=" * 78)
print("PB+低波动 中性化前后 IC（月度，618只，2015-2026）")
print("=" * 78)
print(f"{'因子':<22}{'IC均值':>10}{'|t|':>8}{'ICIR':>8}")
print("-" * 78)
ic_store = {}
for name, fn in FACTORS.items():
    ics = [spearman(fn(i), fwd_ret(i)) for i in rebal]
    ic_store[name] = ics
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() > 0 else 0
    print(f"{name:<22}{s.mean():>+10.4f}{tstat(ics):>8.2f}{icir:>8.2f}")

# ============ 策略回测 ============
print("\n" + "=" * 78)
print("策略回测（top50，月度，含成本）")
print("=" * 78)
print(f"{'策略':<22}{'累计':>10}{'年化':>10}{'超额':>8}{'夏普':>8}{'换手':>8}{'回撤':>8}")
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
for name, fn in FACTORS.items():
    r = backtest(fn)
    results[name] = r
    print(f"{name:<22}{r[0]:>+9.1f}%{r[1]:>+9.2f}%"
          f"{r[2]:>+7.2f}{r[3]:>8.2f}{r[4]:>7.1f}%{r[5]:>8.2f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

names = list(ic_store.keys())
means = [pd.Series(ic_store[n]).dropna().mean() for n in names]
axes[0].barh(names[::-1], means[::-1])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("PB+低波动 中性化前后 IC")
axes[0].set_xlabel("IC 均值")
axes[0].grid(alpha=0.3)

for name, r in results.items():
    axes[1].plot(r[6].values, linewidth=2, label=name)
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("PB+低波动 中性化前后净值（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=7, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_neutral_combo.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_neutral_combo.png")
