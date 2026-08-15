# -*- coding: utf-8 -*-
"""
④ PB+低波动 基准模型的样本外验证：2021-01-04 切分
检验：PB+低波动 的增量 alpha（含行业中性化）是真实规律，还是 2015 牛市过拟合
对比：原始组合 vs 行业中性组合 vs 双重中性组合 在样本内/外的 IC + 回测
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
SPLIT = "2021-01-04"

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
split_pos = close.index.get_indexer([pd.Timestamp(SPLIT)], method='nearest')[0]
print(f"切分点: {dates[split_pos]}（第 {split_pos} 天 / 共 {len(dates)} 天）")

rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]
print(f"月度换仓期：样本内 {len(in_rebal)} 期，样本外 {len(out_rebal)} 期")

mcap = close.mul(total_share, axis=1)
log_mcap = np.log(mcap.where(mcap > 0))
ind_letter = industry.str.slice(0, 1).fillna('Z')
IND_DUM = {L: (ind_letter == L).astype(float) for L in ind_letter.unique() if L != 'Z'}


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def factor_lowvol(i):
    if i < LOWVOL_WIN:
        return pd.Series(np.nan, index=close.columns)
    window = close.iloc[i - LOWVOL_WIN:i]
    rets = window.pct_change().iloc[1:]
    return rets.std()


def combo_score(i):
    return -zscore(clean(pb.iloc[i])) + (-zscore(winsorize(factor_lowvol(i))))


def neutral_resid(i, y_series, controls):
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


FACTORS = {
    "PB+低波动(原始)": lambda i: combo_score(i),
    "PB+低波动(行业中性)": lambda i: neutral_resid(i, combo_score(i), list(IND_DUM.values())),
    "PB+低波动(双重中性)": lambda i: neutral_resid(i, combo_score(i), [log_mcap] + list(IND_DUM.values())),
}

# ============ IC 分期间 ============
print("\n" + "=" * 78)
print("PB+低波动 IC 分期间：样本内 vs 样本外（月度）")
print("=" * 78)
print(f"{'因子':<22}{'样本内IC':>10}{'样本内|t|':>10}{'样本外IC':>10}{'样本外|t|':>10}")
print("-" * 78)
ic_in_all, ic_out_all = {}, {}
for name, fn in FACTORS.items():
    ics_in = [spearman(fn(i), fwd_ret(i)) for i in in_rebal]
    ics_out = [spearman(fn(i), fwd_ret(i)) for i in out_rebal]
    ic_in_all[name] = pd.Series(ics_in).dropna().mean()
    ic_out_all[name] = pd.Series(ics_out).dropna().mean()
    print(f"{name:<22}{ic_in_all[name]:>+10.4f}{tstat(ics_in):>10.2f}"
          f"{ic_out_all[name]:>+10.4f}{tstat(ics_out):>10.2f}")

# ============ 策略分期间回测 ============
print("\n" + "=" * 78)
print("PB+低波动 分期间回测（top50，月度，含成本）")
print("=" * 78)
print(f"{'因子':<22}{'期间':<8}{'策略年化':>10}{'超额':>8}{'回撤':>8}{'夏普':>8}{'换手':>8}")
print("-" * 78)


def backtest(rebal_list, fn):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal_list:
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
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    return annual, annual - b_annual, dd, sharpe, np.mean(turns) * 100, cum


for name, fn in FACTORS.items():
    a_in, x_in, d_in, s_in, t_in, cum_in = backtest(in_rebal, fn)
    a_out, x_out, d_out, s_out, t_out, cum_out = backtest(out_rebal, fn)
    print(f"{name:<22}{'样本内':<8}{a_in:>+10.2f}{x_in:>+8.2f}{d_in:>8.2f}{s_in:>8.2f}{t_in:>8.1f}")
    print(f"{'':<22}{'样本外':<8}{a_out:>+10.2f}{x_out:>+8.2f}{d_out:>8.2f}{s_out:>8.2f}{t_out:>8.1f}")
    if name == "PB+低波动(行业中性)":
        cum_hold = cum_out  # 留作绘图

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

names = list(ic_in_all.keys())
ins = [ic_in_all[n] for n in names]
outs = [ic_out_all[n] for n in names]
x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, ins, w, label='样本内(2015-2021)', color='steelblue')
axes[0].bar(x + w/2, outs, w, label='样本外(2021-2026)', color='orange')
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, fontsize=7, rotation=10)
axes[0].set_title("PB+低波动 IC：样本内 vs 样本外")
axes[0].set_ylabel("IC 均值")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 行业中性版全程净值
full_cum = pd.concat([backtest(in_rebal, FACTORS["PB+低波动(行业中性)"])[5],
                      backtest(out_rebal, FACTORS["PB+低波动(行业中性)"])[5]]).reset_index(drop=True)
axes[1].plot(full_cum.values, color='green', linewidth=2, label='PB+低波动(行业中性) 含成本')
axes[1].axvline(len(in_rebal), color='red', linewidth=1, linestyle=':', label='样本外分界(2021)')
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("行业中性组合净值（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_lowvol_oos.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_lowvol_oos.png")
