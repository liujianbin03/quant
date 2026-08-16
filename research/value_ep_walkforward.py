# -*- coding: utf-8 -*-
"""
PB / EP / PB+EP 的 walk-forward（滚动 + 多切点）样本外验证
—— 把 value_ep_robustness.py 的单次 2021-01 切分，升级为多个年度切点，
    判定"PB+EP 最稳"是真实规律还是单一切点的偶然。

两个维度：
  1) 滚动 IC 时序：12 期(约1年)滚动平均 IC，看各因子 alpha 的时间结构
     （预期印证：PB 后发、EP 早衰、PB+EP 全程稳）
  2) 多切点 OOS：8 个年度切点(2017~2024)，每个切点做一次 样本内/样本外 检验，
     统计"样本外 IC 方向一致率 + |t|>2 显著率 + 样本外年化>0 率"

口径：分数高=便宜；月度 top50 等权；ROUND_TRIP=0.003；区间 2015-2026。
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
ROLL_WIN = 12          # 滚动 IC 窗口（期数≈月数）


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
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = size['totalShare'].reindex(close.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
rebal_dates = [dates[i] for i in rebal]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def score_pb(i):
    return -zscore(clean(pb.iloc[i]))


def score_ep(i):
    return zscore(winsorize(ep.iloc[i]))


def score_pb_ep(i):
    return score_pb(i) + score_ep(i)


SCORES = {"PB": score_pb, "EP": score_ep, "PB+EP": score_pb_ep}
NAMES = list(SCORES.keys())


def universe_ex_small(i, q=SMALL_PCT):
    mi = mcap.iloc[i]
    thr = mi.quantile(q)
    return mi[mi > thr].index


# ============ 1) 逐期 IC（全样本 + 剔小30%） ============
ic_full = {n: [] for n in NAMES}
ic_ex = {n: [] for n in NAMES}
for i in rebal:
    fwd = fwd_ret(i)
    for n, fn in SCORES.items():
        sc = fn(i)
        ic_full[n].append(spearman(sc, fwd))
        uni = universe_ex_small(i)
        ic_ex[n].append(spearman(sc[uni], fwd[uni]))

ic_full_df = pd.DataFrame(ic_full, index=rebal_dates)
ic_ex_df = pd.DataFrame(ic_ex, index=rebal_dates)

print("=" * 96)
print("逐期 IC 汇总（全样本 vs 剔小30%）")
print("=" * 96)
print(f"{'因子':<8}{'全样本IC':>10}{'全样本|t|':>10}{'剔小30%IC':>11}{'剔小30%|t|':>11}")
print("-" * 96)
for n in NAMES:
    sf = ic_full_df[n].dropna()
    sx = ic_ex_df[n].dropna()
    print(f"{n:<8}{sf.mean():>+10.4f}{abs(tstat(sf)):>10.2f}{sx.mean():>+11.4f}"
          f"{abs(tstat(sx)):>11.2f}")
print("-" * 96)

# ============ 2) 多切点 OOS ============
print("\n" + "=" * 96)
print("多切点样本外检验：8 个年度切点（2017~2024），样本内→样本外")
print("=" * 96)


def split_at(rebal_dates, split_ts):
    """返回 (in_rebal_idx, out_rebal_idx) 按 rebal 序号切分"""
    idx = None
    for k, d in enumerate(rebal_dates):
        if d >= split_ts:
            idx = k
            break
    if idx is None:
        return rebal, []
    return rebal[:idx], rebal[idx:]


def oos_backtest(fn, out_rebal, universe_fn=None):
    rets, prev = [], set()
    for i in out_rebal:
        sc = fn(i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index
        if universe_fn is not None:
            valid = valid.intersection(universe_fn(i))
        if len(valid) >= N_HOLD:
            top = sc[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            turn = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(gross - turn * ROUND_TRIP)
            prev = set(top)
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    return (cum.iloc[-1] ** (12 / len(s)) - 1) * 100 if len(s) else 0.0


split_years = list(range(2017, 2025))  # 2017..2024
results = {n: {"全样本": [], "剔小30%": []} for n in NAMES}

print(f"{'切点':<12}" + "".join([f"{n:>12}" for n in NAMES]) + "   ← 样本外 IC (全样本)")
for yr in split_years:
    split_ts = pd.Timestamp(f"{yr}-01-01")
    in_r, out_r = split_at(rebal_dates, split_ts)
    if len(in_r) < 12 or len(out_r) < 12:
        continue
    line = f"{yr}-01"
    for n, fn in SCORES.items():
        ics_out = [spearman(fn(i), fwd_ret(i)) for i in out_r]
        m_out = pd.Series(ics_out).dropna().mean()
        line += f"{m_out:>+12.4f}"
    print(line)
print()

# 聚合统计
print(f"{'因子':<8}{'口径':<8}{'OOS IC均值':>11}  {'符号一致':>13}  {'|t|>2显著':>13}  {'年化>0':>11}")
print("-" * 96)
for n, fn in SCORES.items():
    for uni_label, uni_fn in [("全样本", None), ("剔小30%", universe_ex_small)]:
        sign_hit, t_hit, ann_hit, oos_ics, oos_anns = 0, 0, 0, [], []
        cnt = 0
        for yr in split_years:
            split_ts = pd.Timestamp(f"{yr}-01-01")
            in_r, out_r = split_at(rebal_dates, split_ts)
            if len(in_r) < 12 or len(out_r) < 12:
                continue
            cnt += 1
            ics_in = [spearman(fn(i), fwd_ret(i)) for i in in_r]
            ics_out = [spearman(fn(i), fwd_ret(i)) for i in out_r]
            if uni_fn is not None:
                ics_in = [spearman(fn(i)[uni_fn(i)], fwd_ret(i)[uni_fn(i)]) for i in in_r]
                ics_out = [spearman(fn(i)[uni_fn(i)], fwd_ret(i)[uni_fn(i)]) for i in out_r]
            m_in = pd.Series(ics_in).dropna().mean()
            m_out = pd.Series(ics_out).dropna().mean()
            t_out = abs(tstat(ics_out))
            ann = oos_backtest(fn, out_r, uni_fn)
            if m_in * m_out > 0:
                sign_hit += 1
            if t_out >= 2:
                t_hit += 1
            if ann > 0:
                ann_hit += 1
            oos_ics.append(m_out)
            oos_anns.append(ann)
        results[n][uni_label] = oos_anns
        print(f"{n:<8}{uni_label:<8}{np.mean(oos_ics):>+11.4f}"
              f"   {sign_hit}/{cnt}={sign_hit/cnt*100:>3.0f}%"
              f"   {t_hit}/{cnt}={t_hit/cnt*100:>3.0f}%"
              f"   {ann_hit}/{cnt}={ann_hit/cnt*100:>3.0f}%")
    print("-" * 96)

# ============ 图 ============
fig, axes = plt.subplots(2, 2, figsize=(17, 10))

colors = {"PB": "gray", "EP": "steelblue", "PB+EP": "crimson"}
for n in NAMES:
    axes[0, 0].plot(ic_full_df.index, ic_full_df[n].rolling(ROLL_WIN).mean(),
                    linewidth=2, color=colors[n], label=n)
axes[0, 0].axhline(0, color='black', linewidth=0.8)
axes[0, 0].axvline(pd.Timestamp("2021-01-01"), color='red', linewidth=1, linestyle=':', alpha=0.7)
axes[0, 0].set_title(f"滚动 {ROLL_WIN} 期 IC 均值（全样本）")
axes[0, 0].set_ylabel("IC")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

for n in NAMES:
    axes[0, 1].plot(ic_ex_df.index, ic_ex_df[n].rolling(ROLL_WIN).mean(),
                    linewidth=2, color=colors[n], label=n)
axes[0, 1].axhline(0, color='black', linewidth=0.8)
axes[0, 1].axvline(pd.Timestamp("2021-01-01"), color='red', linewidth=1, linestyle=':', alpha=0.7)
axes[0, 1].set_title(f"滚动 {ROLL_WIN} 期 IC 均值（剔小30%）")
axes[0, 1].set_ylabel("IC")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# OOS 年化（多切点）
x = np.arange(len(split_years))
w = 0.27
for j, n in enumerate(NAMES):
    axes[1, 0].bar(x + (j - 1) * w, results[n]["全样本"], w, color=colors[n], label=n)
axes[1, 0].axhline(0, color='black', linewidth=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f"{y}" for y in split_years])
axes[1, 0].set_title("各切点 样本外年化（全样本，切点→2026）")
axes[1, 0].set_ylabel("年化 (%)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, axis='y')

for j, n in enumerate(NAMES):
    axes[1, 1].bar(x + (j - 1) * w, results[n]["剔小30%"], w, color=colors[n], label=n)
axes[1, 1].axhline(0, color='black', linewidth=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([f"{y}" for y in split_years])
axes[1, 1].set_title("各切点 样本外年化（剔小30%，切点→2026）")
axes[1, 1].set_ylabel("年化 (%)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figures/value_ep_walkforward.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_ep_walkforward.png")
