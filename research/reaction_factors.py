# -*- coding: utf-8 -*-
"""
A：市场反应因子增量检验 —— 新闻/政策的"果"是否在价值之外提供增量 alpha
思路：不直接读新闻（价格已消化、历史数据不干净、易过拟合），
     而是读「市场对新闻的反应」留下的量价痕迹，与慢因子正交：

  - 换手率      ：1月均换手（关注度，A股高换手→低未来收益）
  - 异常换手    ：1月换手 / 6月均换手（事件/关注度突增）
  - MAX彩票效应 ：1月最大单日涨幅（散户博彩偏好→低未来收益，Bali et al. 2011）
  - 涨停次数    ：近3月日涨幅≥9.5%的天数（事件驱动/投机热度，用close近似）

检验：每个反应因子 ① 单独 IC ② 叠加到 PB+EP 上的增量 IC + 回测
口径：分数高=好（低换手/低MAX/少涨停/便宜）；月度 top20；含成本；剔小30%。
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

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126          # 6个月暖机，保证各因子有足够历史


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
    si = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
daily_ret = close.pct_change(fill_method=None)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]
nan_s = lambda: pd.Series(np.nan, index=close.columns)


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def universe_ex_small(i):
    mi = mcap.iloc[i]
    thr = mi.quantile(SMALL_PCT)
    return mi[mi > thr].index


# ============ 因子（原始值，分数高=好） ============
def score_pb_ep(i):
    return -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))


def score_turn(i):
    """低换手好"""
    if i < 21:
        return nan_s()
    return -zscore(winsorize(turn.iloc[i - 21:i].mean()))


def score_abn_turn(i):
    """异常换手低好（1月换手 / 6月均换手）"""
    if i < WARMUP:
        return nan_s()
    t1 = turn.iloc[i - 21:i].mean()
    t6 = turn.iloc[i - WARMUP:i].mean()
    return -zscore(winsorize(t1 / t6.replace(0, np.nan)))


def score_max(i):
    """低MAX好（1月最大单日涨幅）"""
    if i < 21:
        return nan_s()
    return -zscore(winsorize(daily_ret.iloc[i - 21:i].max()))


def score_limit_cnt(i):
    """少涨停好（近3月日涨幅>=9.5%的天数）"""
    if i < 63:
        return nan_s()
    return -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))


BASE = {"PB+EP": score_pb_ep}
REACT = {
    "换手率": score_turn,
    "异常换手": score_abn_turn,
    "MAX彩票": score_max,
    "涨停次数": score_limit_cnt,
}


def combo(base_fn, react_fn):
    def fn(i):
        return base_fn(i) + react_fn(i)
    return fn


# ============ IC 分析 ============
print("=" * 96)
print("反应因子 IC（单独 vs 叠加到 PB+EP 的增量），月度，剔小30%前")
print("=" * 96)
print(f"{'因子':<18}{'单独IC':>10}{'|t|':>8}   {'PB+EP+该因子 IC':>16}{'|t|':>8}")
print("-" * 96)

ic_store = {}
base_ics = [spearman(score_pb_ep(i), fwd_ret(i)) for i in rebal]
sb = pd.Series(base_ics).dropna()
print(f"{'PB+EP(基准)':<18}{sb.mean():>+10.4f}{tstat(base_ics):>8.2f}")
for name, rfn in REACT.items():
    ics_alone = [spearman(rfn(i), fwd_ret(i)) for i in rebal]
    ic_store[name] = ics_alone
    sa = pd.Series(ics_alone).dropna()
    cf = combo(score_pb_ep, rfn)
    ics_combo = [spearman(cf(i), fwd_ret(i)) for i in rebal]
    sc = pd.Series(ics_combo).dropna()
    print(f"{name:<18}{sa.mean():>+10.4f}{tstat(ics_alone):>8.2f}   "
          f"{sc.mean():>+16.4f}{tstat(ics_combo):>8.2f}")
print("-" * 96)

# ============ 回测 ============
print("\n" + "=" * 96)
print("策略回测（top20 月度含成本，剔小30%）—— 反应因子是否在 PB+EP 上增量")
print("=" * 96)


def backtest(fn):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal:
        sc = fn(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index.intersection(universe_ex_small(i))
        bench.append(fwd[valid].mean())
        if len(valid) >= N_HOLD:
            top = sc[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(gross - turnv * ROUND_TRIP)
            turns.append(turnv)
            prev = set(top)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    b = pd.Series(bench)
    b_cum = (1 + b).cumprod()
    b_annual = (b_cum.iloc[-1] ** (12 / len(b)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, annual - b_annual, sharpe, mdd, np.mean(turns) * 100, cum


print(f"{'策略':<20}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}{'换手':>7}")
print("-" * 96)
results = {}
r0 = backtest(score_pb_ep)
results["PB+EP(基准)"] = r0
print(f"{'PB+EP(基准)':<20}{r0[0]:>+8.2f}%{r0[1]:>+7.2f}{r0[2]:>7.2f}{r0[3]:>7.2f}%{r0[4]:>6.1f}%")
for name, rfn in REACT.items():
    r = backtest(combo(score_pb_ep, rfn))
    results[f"PB+EP+{name}"] = r
    print(f"{'PB+EP+'+name:<20}{r[0]:>+8.2f}%{r[1]:>+7.2f}{r[2]:>7.2f}{r[3]:>7.2f}%{r[4]:>6.1f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

names = list(REACT.keys())
alone = [pd.Series(ic_store[n]).dropna().mean() for n in names]
combo_ic = [pd.Series([spearman(combo(score_pb_ep, REACT[n])(i), fwd_ret(i)) for i in rebal]).dropna().mean() for n in names]
x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, alone, w, label='单独', color='steelblue')
axes[0].bar(x + w/2, combo_ic, w, label='PB+EP+该因子', color='darkorange')
axes[0].axhline(sb.mean(), color='crimson', linewidth=1, linestyle='--', label=f'PB+EP基准 {sb.mean():.3f}')
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, fontsize=8)
axes[0].set_title("反应因子 IC：单独 vs 叠加")
axes[0].set_ylabel("IC 均值")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, axis='y')

for name, r in results.items():
    axes[1].plot(r[5].values, linewidth=2, label=name)
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("净值：PB+EP vs 叠加反应因子（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/reaction_factors.png", dpi=150)
print("\n[OK] 图表已保存 figures/reaction_factors.png")
