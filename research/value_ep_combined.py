# -*- coding: utf-8 -*-
"""
终极检验：退市股修正 + 剔小30%市值 同时做 —— PB / EP / PB+EP 谁站得住

此前两个抗性是分开测的：
  - 退市修正（value_ep_robustness.py Part1）：全样本，无剔小30%（退市股缺股本）
  - 剔小30%（ep_factor.py / value_ep_combo.py）：仅现存股，无退市股
本脚本用 fetch_delisted_share.py 抓到的退市股股本，把两者合并：
  1) 股票池 = 现存 618 + 退市股抽样(每8取1 ≈ 真实退市占比 4.9%)
  2) 市值 = close × totalShare（现存股 + 退市股都有股本）
  3) 每个换仓点剔除市值最小 30%
  4) 持有期退市 fwd=-90%，统计踩雷

结论口径：分数高=便宜；月度 top50 等权；ROUND_TRIP=0.003。
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
DELISTED_CACHE = "delisted_cache.pkl"
DELISTED_VAL_CACHE = "delisted_val_cache.pkl"
DELISTED_SHARE_CACHE = "delisted_share_cache.pkl"

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
DELIST_LOSS = -0.9
SMALL_PCT = 0.30
SAMPLE_STEP = 8       # 退市股抽样步长（每8取1 ≈ 真实 4.9%）


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


def make_scores(pe_df, pb_df):
    with np.errstate(divide='ignore', invalid='ignore'):
        ep_df = (1.0 / pe_df).replace([np.inf, -np.inf], np.nan)

    def score_pb(i):
        return -zscore(clean(pb_df.iloc[i]))

    def score_ep(i):
        return zscore(winsorize(ep_df.iloc[i]))

    def score_pb_ep(i):
        return score_pb(i) + score_ep(i)

    return {"PB": score_pb, "EP": score_ep, "PB+EP": score_pb_ep}


def backtest(close_df, score_fn, rebal, mcap_df=None, delist_loss=None, ex_small=False):
    rets, bench, turns, picks = [], [], [], 0
    prev = set()
    for i in rebal:
        fwd = close_df.iloc[i + HOLD] / close_df.iloc[i] - 1
        dmask = None
        if delist_loss is not None:
            dmask = close_df.iloc[i].notna() & close_df.iloc[i + HOLD].isna()
            fwd = fwd.copy()
            fwd[dmask] = delist_loss
        sc = score_fn(i)
        valid = sc.dropna().index
        if ex_small and mcap_df is not None:
            mi = mcap_df.iloc[i]
            thr = mi.quantile(SMALL_PCT)
            valid = valid.intersection(mi[mi > thr].index)
        bench.append(fwd[valid].mean())
        if len(valid) >= N_HOLD:
            top = sc[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            if dmask is not None:
                picks += int(dmask[top].sum())
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
    return annual, annual - b_annual, sharpe, mdd, np.mean(turns) * 100, picks, cum


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    size = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)
with open(DELISTED_VAL_CACHE, 'rb') as f:
    dval = pickle.load(f)
with open(DELISTED_SHARE_CACHE, 'rb') as f:
    dshare = pickle.load(f)

close_l = listed['close']
pe_l = val['pe'].reindex(close_l.index)
pb_l = val['pb'].reindex(close_l.index)
share_l = size['totalShare'].reindex(close_l.columns)

close_d = delisted['close']
pe_d = dval['pe'].reindex(close_d.index)
pb_d = dval['pb'].reindex(close_d.index)

print(f"退市股股本覆盖率: {dshare.notna().sum()}/{len(dshare)}", flush=True)
share_d = dshare.reindex(close_d.columns)  # 退市股股本（快照）

# 退市股抽样（每8取1）
sample = list(close_d.columns)[::SAMPLE_STEP]
cd = close_d[sample].reindex(close_l.index)
pe_dd = pe_d[sample].reindex(close_l.index)
pb_dd = pb_d[sample].reindex(close_l.index)
sd = share_d[sample]

# 合并
close_all = pd.concat([close_l, cd], axis=1)
pe_all = pd.concat([pe_l, pe_dd], axis=1)
pb_all = pd.concat([pb_l, pb_dd], axis=1)
share_all = pd.concat([share_l, sd])
mcap_all = close_all.mul(share_all, axis=1)

dates = close_all.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
scores = make_scores(pe_all, pb_all)

ratio = len(sample) / (close_l.shape[1] + len(sample)) * 100
print(f"合并股票池: {close_all.shape[1]} 只（现存618 + 退市{len(sample)}，退市占比 {ratio:.1f}%）")

# ============ 四象限对比 ============
print("\n" + "=" * 96)
print("终极检验：退市修正(损失-90%) × 剔小30%，四象限对比")
print("=" * 96)
print(f"{'因子':<10}{'退市修正':<8}{'剔小30%':<8}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}{'换手':>7}{'踩雷':>6}")
print("-" * 96)

results = {}
for n in ["PB", "EP", "PB+EP"]:
    fn = scores[n]
    # 1) 无退市修正 + 无剔小30%（基线）
    r00 = backtest(close_all, fn, rebal)
    # 2) 退市修正 + 无剔小30%
    r10 = backtest(close_all, fn, rebal, delist_loss=DELIST_LOSS)
    # 3) 无退市修正 + 剔小30%
    r01 = backtest(close_all, fn, rebal, mcap_df=mcap_all, ex_small=True)
    # 4) 退市修正 + 剔小30%（终极）
    r11 = backtest(close_all, fn, rebal, mcap_df=mcap_all, delist_loss=DELIST_LOSS, ex_small=True)
    results[n] = {"00": r00, "10": r10, "01": r01, "11": r11}
    print(f"{n:<10}{'否':<8}{'否':<8}{r00[0]:>+8.2f}%{r00[1]:>+7.2f}{r00[2]:>7.2f}{r00[3]:>7.2f}%{r00[4]:>6.1f}%{r00[5]:>6}")
    print(f"{'':<10}{'是':<8}{'否':<8}{r10[0]:>+8.2f}%{r10[1]:>+7.2f}{r10[2]:>7.2f}{r10[3]:>7.2f}%{r10[4]:>6.1f}%{r10[5]:>6}")
    print(f"{'':<10}{'否':<8}{'是':<8}{r01[0]:>+8.2f}%{r01[1]:>+7.2f}{r01[2]:>7.2f}{r01[3]:>7.2f}%{r01[4]:>6.1f}%{r01[5]:>6}")
    print(f"{'':<10}{'是':<8}{'是':<8}{r11[0]:>+8.2f}%{r11[1]:>+7.2f}{r11[2]:>7.2f}{r11[3]:>7.2f}%{r11[4]:>6.1f}%{r11[5]:>6}")
    print("-" * 96)

# ============ 图 ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for j, n in enumerate(["PB", "EP", "PB+EP"]):
    ax = axes[j]
    ax.plot(results[n]["00"][6].values, linewidth=2, color='gray', linestyle=':', label='无修正无剔小')
    ax.plot(results[n]["10"][6].values, linewidth=2, color='steelblue', label='退市修正')
    ax.plot(results[n]["01"][6].values, linewidth=2, color='darkorange', linestyle='--', label='剔小30%')
    ax.plot(results[n]["11"][6].values, linewidth=2, color='crimson', label='退市修正+剔小30%')
    ax.axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_title(f"{n}：四象限净值（含成本）")
    ax.set_xlabel("换仓期")
    ax.set_ylabel("累计净值")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_ep_combined.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_ep_combined.png")
