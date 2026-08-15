# -*- coding: utf-8 -*-
"""
EP（盈利收益率）因子 + 剔除最小 30% 市值 —— 依据 Liu, Stambaugh & Yuan (2019)
"Size and Value in China"（CH-3 三因子）

该文的两个核心结论，直接针对本项目现有做法：
  1) 中国市场 EP（=1/市盈率TTM）比 PB/BM 更强：账面价值会被 A 股小市值股的
     "壳价值"污染，导致 PB 因子失真，而 EP 更干净。
  2) 应剔除市值最小的 30% 股票：最尾部的小票是壳/退市噪声而非 alpha，
     与本项目"面值过滤救命 / 退市股是反转杀手"的发现同源。

本脚本用与 value_factor.py / value_lowvol_reversal.py 完全相同的框架：
  - 因子分数方向统一为「分数越高越便宜」（好因子 IC>0）
  - 月度调仓，top50 等权，含成本 ROUND_TRIP=0.003，区间 2015-2026
  - 市值 = close × totalShare（totalShare 为快照式，股本变动慢，对横截面排序影响小）
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
SMALL_PCT = 0.30          # 剔除市值最小 30%（Liu-Stambaugh-Yuan 口径）


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def clean(f):
    """估值因子清洗：非正(亏损)置 NaN，再 1%~99% 缩尾"""
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


def winsorize(s, q=0.01):
    """全样本缩尾（允许负值，用于含亏损样本的 EP）"""
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

# EP = 1 / 市盈率TTM
with np.errstate(divide='ignore', invalid='ignore'):
    ep = 1.0 / pe
ep = ep.replace([np.inf, -np.inf], np.nan)

# 市值（快照股本 × 价格，横截面随时间变化）
mcap = close.mul(total_share, axis=1)

dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


# ============ 因子分数（高=便宜/好） ============
def score_pb(i):
    """低 PB 好：取负 zscore"""
    return -zscore(clean(pb.iloc[i]))


def score_ep_pos(i):
    """高 EP 好（仅正收益样本，与 PB 同口径可比）"""
    return zscore(clean(ep.iloc[i]))


def score_ep_all(i):
    """高 EP 好（含亏损样本：亏损股 EP<0，自然垫底）"""
    return zscore(winsorize(ep.iloc[i]))


FACTORS = {
    "PB(低好)":        score_pb,
    "EP正收益(高好)":  score_ep_pos,
    "EP全样本(高好)":  score_ep_all,
}


def universe_ex_small(i, q=SMALL_PCT):
    """剔除市值最小 q 的股票，返回保留的代码 Index"""
    mi = mcap.iloc[i]
    thr = mi.quantile(q)
    return mi[mi > thr].index


# ============ IC 分析 ============
print("=" * 90)
print("EP vs PB 因子 IC（月度，618只，2015-2026）—— 检验 Liu-Stambaugh-Yuan 结论①")
print("=" * 90)
print(f"{'因子':<18}{'全样本IC':>10}{'|t|':>8}{'ICIR':>8}   {'剔小30%后IC':>12}{'|t|':>8}{'ICIR':>8}")
print("-" * 90)

ic_store = {}
for name, fn in FACTORS.items():
    ics_full, ics_ex = [], []
    for i in rebal:
        sc = fn(i)
        fwd = fwd_ret(i)
        ics_full.append(spearman(sc, fwd))
        uni = universe_ex_small(i)
        ics_ex.append(spearman(sc[uni], fwd[uni]))
    ic_store[name] = ics_full
    sf = pd.Series(ics_full).dropna()
    sx = pd.Series(ics_ex).dropna()
    icir_f = sf.mean() / sf.std() if sf.std() > 0 else 0
    icir_x = sx.mean() / sx.std() if sx.std() > 0 else 0
    print(f"{name:<18}{sf.mean():>+10.4f}{tstat(ics_full):>8.2f}{icir_f:>8.2f}   "
          f"{sx.mean():>+12.4f}{tstat(ics_ex):>8.2f}{icir_x:>8.2f}")
print("-" * 90)
print("注：分数高=便宜，故「IC>0 且 |t|>2」= 有效（便宜→高收益）。")

# ============ 策略回测 ============
print("\n" + "=" * 90)
print("策略回测（top50，月度，含成本）—— 检验 EP 是否强于 PB + 剔除小30%是否提升")
print("=" * 90)


def backtest(score_fn, universe_fn=None):
    """返回 (累计%, 年化%, 超额pp, 夏普, 换手%, 回撤%, 净值Series)"""
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


print(f"{'策略':<24}{'累计':>9}{'年化':>9}{'超额':>8}{'夏普':>7}{'换手':>7}{'回撤':>8}")
print("-" * 90)

results = {}
pb_full = backtest(score_pb)
results["PB(基准)"] = pb_full
print(f"{'PB(全样本)':<24}{pb_full[0]:>+8.1f}%{pb_full[1]:>+8.2f}%{pb_full[2]:>+7.2f}"
      f"{pb_full[3]:>7.2f}{pb_full[4]:>6.1f}%{pb_full[5]:>7.2f}%")

ep_pos_full = backtest(score_ep_pos)
results["EP正收益(全样本)"] = ep_pos_full
print(f"{'EP正收益(全样本)':<24}{ep_pos_full[0]:>+8.1f}%{ep_pos_full[1]:>+8.2f}%"
      f"{ep_pos_full[2]:>+7.2f}{ep_pos_full[3]:>7.2f}{ep_pos_full[4]:>6.1f}%"
      f"{ep_pos_full[5]:>7.2f}%")

ep_all_full = backtest(score_ep_all)
results["EP全样本(全样本)"] = ep_all_full
print(f"{'EP全样本(全样本)':<24}{ep_all_full[0]:>+8.1f}%{ep_all_full[1]:>+8.2f}%"
      f"{ep_all_full[2]:>+7.2f}{ep_all_full[3]:>7.2f}{ep_all_full[4]:>6.1f}%"
      f"{ep_all_full[5]:>7.2f}%")

print("-" * 90)

pb_ex = backtest(score_pb, universe_ex_small)
results["PB(剔小30%)"] = pb_ex
print(f"{'PB(剔小30%)':<24}{pb_ex[0]:>+8.1f}%{pb_ex[1]:>+8.2f}%{pb_ex[2]:>+7.2f}"
      f"{pb_ex[3]:>7.2f}{pb_ex[4]:>6.1f}%{pb_ex[5]:>7.2f}%")

ep_pos_ex = backtest(score_ep_pos, universe_ex_small)
results["EP正收益(剔小30%)"] = ep_pos_ex
print(f"{'EP正收益(剔小30%)':<24}{ep_pos_ex[0]:>+8.1f}%{ep_pos_ex[1]:>+8.2f}%"
      f"{ep_pos_ex[2]:>+7.2f}{ep_pos_ex[3]:>7.2f}{ep_pos_ex[4]:>6.1f}%"
      f"{ep_pos_ex[5]:>7.2f}%")

ep_all_ex = backtest(score_ep_all, universe_ex_small)
results["EP全样本(剔小30%)"] = ep_all_ex
print(f"{'EP全样本(剔小30%)':<24}{ep_all_ex[0]:>+8.1f}%{ep_all_ex[1]:>+8.2f}%"
      f"{ep_all_ex[2]:>+7.2f}{ep_all_ex[3]:>7.2f}{ep_all_ex[4]:>6.1f}%"
      f"{ep_all_ex[5]:>7.2f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# (1) IC 柱状图（全样本 vs 剔小30%）
names = list(FACTORS.keys())
means_full = [pd.Series(ic_store[n]).dropna().mean() for n in names]
x = np.arange(len(names))
w = 0.38
axes[0].barh(x + w / 2, means_full, height=w, color='steelblue', label='全样本')
ic_ex_store = {}
for name, fn in FACTORS.items():
    ic_ex_store[name] = [spearman(fn(i)[universe_ex_small(i)], fwd_ret(i)[universe_ex_small(i)]) for i in rebal]
means_ex = [pd.Series(ic_ex_store[n]).dropna().mean() for n in names]
axes[0].barh(x - w / 2, means_ex, height=w, color='darkorange', label='剔小30%')
axes[0].set_yticks(x)
axes[0].set_yticklabels(names)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("因子 IC（>0 且 |t|>2 = 有效）")
axes[0].set_xlabel("IC 均值")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, axis='x')

# (2) 净值：PB vs EP（全样本）
axes[1].plot(pb_full[6].values, linewidth=2, color='gray', label='PB(基准)')
axes[1].plot(ep_pos_full[6].values, linewidth=2, color='steelblue', label='EP正收益')
axes[1].plot(ep_all_full[6].values, linewidth=2, color='crimson', label='EP全样本')
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("净值：PB vs EP（全样本，含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# (3) 净值：剔小30% 前后
axes[2].plot(ep_pos_full[6].values, linewidth=2, color='steelblue', linestyle='--', label='EP正收益·全样本')
axes[2].plot(ep_pos_ex[6].values, linewidth=2, color='darkorange', label='EP正收益·剔小30%')
axes[2].plot(ep_all_ex[6].values, linewidth=2, color='crimson', label='EP全样本·剔小30%')
axes[2].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[2].set_title("净值：剔除最小30%市值前后（EP）")
axes[2].set_xlabel("换仓期")
axes[2].set_ylabel("累计净值")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/ep_factor.png", dpi=150)
print("\n[OK] 图表已保存 figures/ep_factor.png")
