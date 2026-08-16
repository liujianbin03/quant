# -*- coding: utf-8 -*-
"""
⑥ PB+低波动 基准模型参数微调：样本内选参，样本外验证
- 参数网格：低波动窗口 {60,120,250} × 持仓数 {50,100} × 加权 {等权, IC加权}
- 严谨做法：样本内(2015-2021)选最优，样本外(2021-2026)验证是否仍成立
- 基准模型 = 行业中性 + PB + 低波动
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

HOLD = 21
ROUND_TRIP = 0.0030
IC_WIN = 24  # IC 加权的滚动窗口（期）


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


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_IND_CACHE, 'rb') as f:
    si = pickle.load(f)

close = listed['close']
pb = val['pb'].reindex(close.index)
industry = si['industry'].reindex(close.columns)

dates = close.index
split_pos = close.index.get_indexer([pd.Timestamp(SPLIT)], method='nearest')[0]
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]

ind_letter = industry.str.slice(0, 1).fillna('Z')
IND_DUM = {L: (ind_letter == L).astype(float) for L in ind_letter.unique() if L != 'Z'}


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def factor_lowvol(i, win):
    if i < win:
        return pd.Series(np.nan, index=close.columns)
    window = close.iloc[i - win:i]
    return window.pct_change().iloc[1:].std()


def pb_score(i):
    return -zscore(clean(pb.iloc[i]))


def lowvol_score(i, win):
    return -zscore(winsorize(factor_lowvol(i, win)))


def neutral_resid(i, y_series):
    y = y_series
    X = pd.DataFrame([c for c in IND_DUM.values()]).T.assign(const=1.0)
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


def combo(i, win, wpb, wlv):
    """行业中性化后的加权组合得分"""
    s = wpb * pb_score(i) + wlv * lowvol_score(i, win)
    return neutral_resid(i, s)


# ============ 预计算各窗口的因子得分序列 ============
# 对每个换仓点预存 pb_score 和 lowvol_score，加速参数扫描
def precompute():
    pb_scores = {i: pb_score(i) for i in rebal}
    lv_scores = {win: {i: lowvol_score(i, win) for i in rebal} for win in [60, 120, 250]}
    fwd = {i: fwd_ret(i) for i in rebal}
    return pb_scores, lv_scores, fwd


print("[1/3] 预计算因子得分...", flush=True)
PB_S, LV_S, FWD = precompute()

# IC 加权：滚动窗口内各因子 IC 的比例作为权重
def rolling_ic_weights(win, ic_window=IC_WIN):
    """对每个换仓点，用过去 ic_window 期的 IC 比例做权重"""
    weights = {}
    pb_ics, lv_ics = [], []
    for k, i in enumerate(rebal):
        pb_ics.append(spearman(PB_S[i], FWD[i]))
        lv_ics.append(spearman(LV_S[win][i], FWD[i]))
        if k >= ic_window:
            wpb = np.nanmean(pb_ics[-ic_window:])
            wlv = np.nanmean(lv_ics[-ic_window:])
            denom = abs(wpb) + abs(wlv)
            if denom > 0:
                weights[i] = (abs(wpb) / denom, abs(wlv) / denom)
            else:
                weights[i] = (0.5, 0.5)
        else:
            weights[i] = (0.5, 0.5)
    return weights


# ============ 回测函数 ============
def backtest(rebal_list, win, weight_fn, n_hold):
    rets, bench, turns = [], [], []
    prev = set()
    for i in rebal_list:
        if weight_fn == "equal":
            wpb, wlv = 0.5, 0.5
        else:
            wpb, wlv = weight_fn[i]
        sc = combo(i, win, wpb, wlv)
        fwd = FWD[i]
        valid = sc.dropna().index
        bench.append(fwd[valid].mean())
        if len(valid) >= n_hold:
            top = sc[valid].nlargest(n_hold).index
            gross = fwd[top].mean()
            turn = 1 - len(set(top) & prev) / n_hold if prev else 1.0
            rets.append(gross - turn * ROUND_TRIP)
            turns.append(turn)
            prev = set(top)
    s = pd.Series(rets)
    if len(s) < 5:
        return np.nan, np.nan, np.nan
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    return annual, sharpe, np.mean(turns) * 100


# ============ 参数扫描 ============
print("[2/3] 参数扫描...", flush=True)
ic_weights = {win: rolling_ic_weights(win) for win in [60, 120, 250]}

rows = []
for win in [60, 120, 250]:
    for n_hold in [50, 100]:
        for wmode in ["equal", "ic"]:
            wf = "equal" if wmode == "equal" else ic_weights[win]
            a_in, s_in, t_in = backtest(in_rebal, win, wf, n_hold)
            a_out, s_out, t_out = backtest(out_rebal, win, wf, n_hold)
            rows.append({
                "窗口": win, "持仓": n_hold,
                "加权": "等权" if wmode == "equal" else "IC加权",
                "样本内年化": a_in, "样本内夏普": s_in,
                "样本外年化": a_out, "样本外夏普": s_out,
                "样本外换手": t_out,
            })

df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print("\n" + "=" * 100)
print("参数扫描结果（样本内选参 → 样本外验证）")
print("=" * 100)
print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

# 找出样本外夏普最优的组合
best = df.loc[df["样本外夏普"].idxmax()]
print("\n[最优] 样本外夏普最高组合：")
print(best.to_string())

# ============ 图 ============
print("[3/3] 绘图...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1：样本内 vs 样本外 年化散点（每个组合一个点）
for _, r in df.iterrows():
    color = 'steelblue' if r["加权"] == "等权" else 'orange'
    marker = 'o' if r["持仓"] == 50 else 's'
    axes[0].scatter(r["样本内年化"], r["样本外年化"], c=color, marker=marker, s=80)
axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
axes[0].axvline(0, color='black', linewidth=0.5, alpha=0.5)
axes[0].set_xlabel("样本内年化(%)")
axes[0].set_ylabel("样本外年化(%)")
axes[0].set_title("参数组合：样本内 vs 样本外（右上角=样本内外都好）")
axes[0].grid(alpha=0.3)

# 图2：按窗口分组的样本外年化
for wmode, color in [("等权", 'steelblue'), ("IC加权", 'orange')]:
    sub = df[df["加权"] == wmode]
    for n, m in [(50, 'o'), (100, 's')]:
        sub2 = sub[sub["持仓"] == n]
        axes[1].plot(sub2["窗口"], sub2["样本外年化"], marker=m, color=color,
                     label=f"{wmode}-持仓{n}", linewidth=2)
axes[1].axhline(0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_xlabel("低波动窗口（交易日）")
axes[1].set_ylabel("样本外年化(%)")
axes[1].set_title("样本外年化 vs 低波动窗口")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/value_tuning.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_tuning.png")
