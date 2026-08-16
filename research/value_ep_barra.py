# -*- coding: utf-8 -*-
"""
Barra CNE5 风格中性化：PB / EP / PB+EP 的 alpha 是不是"风格/行业暴露"的幻觉？
—— 短板 5：把 value_neutral.py 的「市值+行业门类」升级为「市值+行业+Beta」三重中性化，
    并扩展到 EP 与 PB+EP，叠加剔小30%，回答"纯因子"还剩多少 alpha。

方法（逐换仓点横截面 OLS 取残差，残差 zscore）：
  1) 市值中性：因子 对 log(市值) 回归
  2) +行业中性：再对行业门类哑变量（证监会首字母 19 大类）回归
  3) +Beta 中性：再对过去60日个股 Beta（对等权市场）回归  → Barra 风格"纯化"

口径：分数高=便宜；月度 top50 等权；ROUND_TRIP=0.003；Beta 需要 60 日暖机，故跳过前3个月。
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
SMALL_PCT = 0.30
BETA_WIN = 60          # Beta 回溯窗口（交易日）


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
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
industry = si['industry'].reindex(close.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
log_mcap = np.log(mcap.where(mcap > 0))
ind_letter = industry.str.slice(0, 1).fillna('Z')

# 行业哑变量（静态 Series，19 门类）
IND_DUM = {L: (ind_letter == L).astype(float) for L in ind_letter.unique() if L != 'Z'}
IND_DUM_LIST = list(IND_DUM.values())

# Beta：过去60日个股对等权市场的回归系数
daily_ret = close.pct_change(fill_method=None)
mkt_ret = daily_ret.mean(axis=1)


def beta_at(i):
    w = daily_ret.iloc[max(0, i - BETA_WIN):i]
    m = mkt_ret.iloc[max(0, i - BETA_WIN):i]
    m = m - m.mean()
    var_m = (m ** 2).mean()
    if var_m <= 0 or len(m) < 30:
        return pd.Series(np.nan, index=close.columns)
    wd = w - w.mean(axis=0)
    cov = wd.mul(m, axis=0).mean(axis=0)
    return cov / var_m


dates = close.index
rebal = [i for i in range(0, len(dates), HOLD)
         if i + HOLD < len(dates) and i >= BETA_WIN]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ 因子分数（原始） ============
def raw_pb(i):
    return -zscore(clean(pb.iloc[i]))


def raw_ep(i):
    return zscore(winsorize(ep.iloc[i]))


def raw_pb_ep(i):
    return raw_pb(i) + raw_ep(i)


def ols_resid(y, controls):
    """controls: list of Series(index=股票代码)。对 controls 回归取残差并 zscore"""
    X = pd.concat(controls, axis=1)
    X = X.assign(const=1.0)
    valid = y.notna() & X.notna().all(axis=1)
    if valid.sum() < 30:
        return pd.Series(np.nan, index=y.index)
    yv = y[valid].values.astype(float)
    Xv = X[valid].values.astype(float)
    Xv = Xv - Xv.mean(axis=0)
    yv = yv - yv.mean()
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    out = pd.Series(np.nan, index=y.index)
    out[valid] = resid
    return zscore(out)


def make_factor(raw_fn, depth):
    """depth: 0=原始 1=+市值 2=+市值+行业 3=+市值+行业+Beta"""
    def fn(i):
        y = raw_fn(i)
        if depth == 0:
            return y
        ctrls = [log_mcap.iloc[i]]
        if depth >= 2:
            ctrls += IND_DUM_LIST
        if depth >= 3:
            ctrls.append(beta_at(i))
        return ols_resid(y, ctrls)
    return fn


RAW = {"PB": raw_pb, "EP": raw_ep, "PB+EP": raw_pb_ep}
FACTORS = {}
for n, rf in RAW.items():
    FACTORS[f"{n}(原始)"] = make_factor(rf, 0)
    FACTORS[f"{n}(+市值)"] = make_factor(rf, 1)
    FACTORS[f"{n}(+市值行业)"] = make_factor(rf, 2)
    FACTORS[f"{n}(+市值行业Beta)"] = make_factor(rf, 3)


def universe_ex_small(i, q=SMALL_PCT):
    mi = mcap.iloc[i]
    thr = mi.quantile(q)
    return mi[mi > thr].index


# ============ IC ============
print("=" * 96)
print("Barra 风格中性化：PB / EP / PB+EP 的 IC（月度，暖机后 2015-2026）")
print("=" * 96)
print(f"{'因子':<20}{'IC均值':>10}{'|t|':>8}{'ICIR':>8}   备注")
print("-" * 96)
ic_store = {}
for name, fn in FACTORS.items():
    ics = [spearman(fn(i), fwd_ret(i)) for i in rebal]
    ic_store[name] = ics
    s = pd.Series(ics).dropna()
    icir = s.mean() / s.std() if s.std() > 0 else 0
    print(f"{name:<20}{s.mean():>+10.4f}{tstat(ics):>8.2f}{icir:>8.2f}")

# 剔小30% 下的完整中性化（最干净口径）
print("-" * 96)
print("剔小30% + 完整中性化（市值+行业+Beta）：")
for n in RAW:
    fn = FACTORS[f"{n}(+市值行业Beta)"]
    ics = [spearman(fn(i)[universe_ex_small(i)], fwd_ret(i)[universe_ex_small(i)]) for i in rebal]
    s = pd.Series(ics).dropna()
    print(f"  {n:<18}IC {s.mean():+.4f}  |t| {tstat(ics):.2f}")

# ============ 回测（原始 vs 完整中性） ============
print("\n" + "=" * 96)
print("策略回测（top50 月度含成本）：原始 vs 完整中性(市值+行业+Beta)")
print("=" * 96)


def backtest(fn, universe_fn=None):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal:
        sc = fn(i)
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
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, annual - b_annual, sharpe, mdd, cum


print(f"{'因子':<10}{'版本':<16}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}")
print("-" * 96)
bt_results = {}
for n in RAW:
    r_raw = backtest(FACTORS[f"{n}(原始)"])
    r_neu = backtest(FACTORS[f"{n}(+市值行业Beta)"])
    r_neu_ex = backtest(FACTORS[f"{n}(+市值行业Beta)"], universe_ex_small)
    bt_results[(n, "原始")] = r_raw
    bt_results[(n, "中性")] = r_neu
    bt_results[(n, "中性+剔小30%")] = r_neu_ex
    print(f"{n:<10}{'原始':<16}{r_raw[0]:>+8.2f}%{r_raw[1]:>+7.2f}{r_raw[2]:>7.2f}{r_raw[3]:>7.2f}%")
    print(f"{'':<10}{'中性':<16}{r_neu[0]:>+8.2f}%{r_neu[1]:>+7.2f}{r_neu[2]:>7.2f}{r_neu[3]:>7.2f}%")
    print(f"{'':<10}{'中性+剔小30%':<16}{r_neu_ex[0]:>+8.2f}%{r_neu_ex[1]:>+7.2f}"
          f"{r_neu_ex[2]:>7.2f}{r_neu_ex[3]:>7.2f}%")
    print("-" * 96)

# ============ 图 ============
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# (0,0) 原始 vs 完整中性 IC
names = list(RAW.keys())
x = np.arange(len(names))
w = 0.35
raw_ic = [pd.Series(ic_store[f"{n}(原始)"]).dropna().mean() for n in names]
neu_ic = [pd.Series(ic_store[f"{n}(+市值行业Beta)"]).dropna().mean() for n in names]
axes[0, 0].bar(x - w / 2, raw_ic, w, label='原始', color='steelblue')
axes[0, 0].bar(x + w / 2, neu_ic, w, label='市值+行业+Beta中性', color='darkorange')
axes[0, 0].axhline(0, color='black', linewidth=0.8)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(names)
axes[0, 0].set_title("IC：原始 vs 完整中性化（alpha 是否纯因子）")
axes[0, 0].set_ylabel("IC 均值")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3, axis='y')

# (0,1) 中性化深度 → IC
depths = ["原始", "+市值", "+市值行业", "+市值行业Beta"]
for n in names:
    vals = [pd.Series(ic_store[f"{n}({d})"]).dropna().mean() for d in depths]
    axes[0, 1].plot(range(len(depths)), vals, marker='o', linewidth=2, label=n)
axes[0, 1].axhline(0, color='black', linewidth=0.8)
axes[0, 1].set_xticks(range(len(depths)))
axes[0, 1].set_xticklabels(depths, fontsize=8)
axes[0, 1].set_title("中性化深度 → IC（越往下越纯）")
axes[0, 1].set_ylabel("IC 均值")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# (1,0) 净值：原始 vs 中性
for n in names:
    axes[1, 0].plot(bt_results[(n, "原始")][4].values, linewidth=2, color='gray', linestyle='--', label=f"{n}原始")
    axes[1, 0].plot(bt_results[(n, "中性")][4].values, linewidth=2, label=f"{n}中性")
axes[1, 0].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1, 0].set_title("净值：原始 vs 完整中性（含成本）")
axes[1, 0].set_xlabel("换仓期")
axes[1, 0].set_ylabel("累计净值")
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(alpha=0.3)

# (1,1) PB+EP：中性 vs 中性+剔小30%
pbep_neu = bt_results[("PB+EP", "中性")][4]
pbep_ex = bt_results[("PB+EP", "中性+剔小30%")][4]
axes[1, 1].plot(pbep_neu.values, linewidth=2, color='crimson', label='PB+EP 中性')
axes[1, 1].plot(pbep_ex.values, linewidth=2, color='darkorange', label='PB+EP 中性+剔小30%')
axes[1, 1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1, 1].set_title("PB+EP：完整中性化后，剔小30% 是否仍提升")
axes[1, 1].set_xlabel("换仓期")
axes[1, 1].set_ylabel("累计净值")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_ep_barra.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_ep_barra.png")
