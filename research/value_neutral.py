# -*- coding: utf-8 -*-
"""
② 市值/行业中性化：验证 PB 的 alpha 是"真价值"还是"藏着小盘/行业 tilt"

方法（逐换仓点横截面）：
  市值中性化：PB 得分 对 log(市值) 回归取残差
  行业中性化：PB 得分 对行业哑变量回归取残差（证监会门类，首字母 19 大类）
  双重中性化：对 log(市值) + 行业哑变量 同时回归取残差

对比：纯PB / 市值中性PB / 行业中性PB / 双重中性PB 的 IC + 回测
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


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def clean(f):
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


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

# 市值 = close × totalShare（快照股本）；行业门类 = 证监会行业首字母
mcap = close.mul(total_share, axis=1)
log_mcap = np.log(mcap.where(mcap > 0))
ind_letter = industry.str.slice(0, 1).fillna('Z')  # 门类（首字母，19大类）

print(f"行业分布（门类）: {ind_letter.value_counts().shape[0]} 类")


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def raw_score(i):
    return -zscore(clean(pb.iloc[i]))


def neutral_resid(i, controls):
    """对横截面得分做 OLS，返回残差（已 zscore）
    control 可为 DataFrame(index=dates) 取 .iloc[i]，或 Series(index=columns) 直接用"""
    y = raw_score(i)
    X = []
    for c in controls:
        if isinstance(c, pd.DataFrame):
            x = c.iloc[i].astype(float)
        else:  # Series，index 是股票代码
            x = c.astype(float)
        X.append(x)
    X = pd.DataFrame(X).T
    X = X.assign(const=1.0)
    valid = y.notna() & X.notna().all(axis=1)
    yv = y[valid].values.astype(float)
    Xv = X[valid].values.astype(float)
    if len(yv) < 30:
        return pd.Series(np.nan, index=y.index)
    # 去均值（去掉常数项共线性）
    Xv = Xv - Xv.mean(axis=0)
    yv = yv - yv.mean()
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    out = pd.Series(np.nan, index=y.index)
    out[valid] = resid
    return zscore(out)


# 行业哑变量（横截面 OLS，用行业均值替代做中性化等价于对门类去均值）
def ind_dummies():
    letters = ind_letter.unique()
    dummies = {}
    for L in letters:
        if L == 'Z':
            continue
        dummies[L] = (ind_letter == L).astype(float)
    return dummies


IND_DUM = ind_dummies()

FACTORS = {
    "PB(原始)": lambda i: raw_score(i),
    "PB(市值中性)": lambda i: neutral_resid(i, [log_mcap]),
    "PB(行业中性)": lambda i: neutral_resid(i, list(IND_DUM.values())),
    "PB(双重中性)": lambda i: neutral_resid(i, [log_mcap] + list(IND_DUM.values())),
}

# ============ IC 分析 ============
print("\n" + "=" * 78)
print("中性化前后 PB 因子 IC（月度，618只，2015-2026）")
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

# ============ 策略回测 ============
print("\n" + "=" * 78)
print("策略回测（top50，月度，含成本）—— 中性化是否保留 alpha")
print("=" * 78)
print(f"{'策略':<16}{'累计':>10}{'年化':>10}{'超额':>8}{'夏普':>8}{'换手':>8}{'回撤':>8}")
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
    print(f"{name:<16}{r[0]:>+9.1f}%{r[1]:>+9.2f}%"
          f"{r[2]:>+7.2f}{r[3]:>8.2f}{r[4]:>7.1f}%{r[5]:>8.2f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

names = list(ic_store.keys())
means = [pd.Series(ic_store[n]).dropna().mean() for n in names]
axes[0].barh(names[::-1], means[::-1])
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("中性化前后 PB 因子 IC")
axes[0].set_xlabel("IC 均值")
axes[0].grid(alpha=0.3)

for name, r in results.items():
    axes[1].plot(r[6].values, linewidth=2, label=name)
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("中性化前后 PB 策略净值（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=7, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_neutral.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_neutral.png")
