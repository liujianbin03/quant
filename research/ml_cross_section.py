# -*- coding: utf-8 -*-
"""
机器学习截面选股（LightGBM walk-forward）—— 短板 3
依据 Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning"：
用 10 个横截面特征预测下月收益，滚动窗口 walk-forward 训练，逐期选 top50。

特征（全部逐换仓点横截面计算，只用当期及之前数据，无前视）：
  价值  ：PB、EP(=1/PE)、PS、PCF
  动量/反转：1月反转、6月动量
  波动/流动：60日波动率、1月均换手率
  风险/规模：60日Beta、log市值

严格口径（套用项目三重检验）：
  - walk-forward：滚动 60 期训练 → 下一期预测（天然样本外，无偷看未来）
  - 含成本：ROUND_TRIP=0.003，计入换手
  - 对比：线性基准 PB+EP（等权），同 universe 同成本
  - 幸存者偏差提示：特征只覆盖现存 618 只，退市股未入池（已知局限，见 README）

输出：ML 预测 IC（样本外）+ ML vs 线性 的回测对比（全样本 / 剔小30%）。
"""
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

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
TRAIN_MIN = 36          # 至少 36 期(3年)才开始预测
ROLL_WIN = 60           # 滚动训练窗口期数(5年)
BETA_WIN = 60


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def winsorize(s, q=0.01):
    s = s.astype(float)
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


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
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
ps = val['ps'].reindex(close.index)
pcf = val['pcf'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
log_mcap = np.log(mcap.where(mcap > 0))
daily_ret = close.pct_change(fill_method=None)
mkt_ret = daily_ret.mean(axis=1)

dates = close.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]


def fwd_ret(i):
    return close.iloc[i + HOLD] / close.iloc[i] - 1


def beta_at(i):
    w = daily_ret.iloc[max(0, i - BETA_WIN):i]
    m = mkt_ret.iloc[max(0, i - BETA_WIN):i]
    m = m - m.mean()
    var_m = (m ** 2).mean()
    if var_m <= 0 or len(m) < 30:
        return pd.Series(np.nan, index=close.columns)
    wd = w - w.mean(axis=0)
    return wd.mul(m, axis=0).mean(axis=0) / var_m


def build_features(i):
    """横截面特征（返回 DataFrame，行=股票，列=特征；每列已缩尾，历史不足则置 NaN）"""
    nan_s = pd.Series(np.nan, index=close.columns)
    f = {}
    f['pb'] = winsorize(pb.iloc[i])
    f['ep'] = winsorize(ep.iloc[i])
    f['ps'] = winsorize(ps.iloc[i])
    f['pcf'] = winsorize(pcf.iloc[i])
    f['rev1m'] = winsorize(close.iloc[i] / close.iloc[i - 21] - 1) if i >= 21 else nan_s
    f['mom6m'] = winsorize(close.iloc[i] / close.iloc[i - 126] - 1) if i >= 126 else nan_s
    f['vol60'] = winsorize(daily_ret.iloc[i - BETA_WIN:i].std()) if i >= BETA_WIN else nan_s
    f['turn1m'] = winsorize(turn.iloc[max(0, i - 21):i].mean()) if i >= 21 else nan_s
    f['logmcap'] = winsorize(log_mcap.iloc[i])
    f['beta60'] = winsorize(beta_at(i))
    return pd.DataFrame(f)


# ============ 预计算特征矩阵 + 目标 ============
print("预计算各换仓期特征矩阵...", flush=True)
X_all, y_all = [], []
for i in rebal:
    X_all.append(build_features(i))
    y_all.append(fwd_ret(i))
FEATURES = list(X_all[0].columns)
print(f"特征 {len(FEATURES)} 个: {FEATURES}")

# ============ walk-forward LightGBM ============
PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
              min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
              random_state=42, n_jobs=-1, verbose=-1)

print(f"\nwalk-forward：滚动{ROLL_WIN}期训练，第{TRAIN_MIN}期起预测（共{len(rebal)-TRAIN_MIN}期）", flush=True)
preds = {}          # test_idx -> 预测收益 Series
for k in range(TRAIN_MIN, len(rebal)):
    lo = max(0, k - ROLL_WIN)
    X_tr = pd.concat(X_all[lo:k])
    y_tr = pd.concat(y_all[lo:k])
    mask = y_tr.notna()
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr, y_tr)
    X_te = X_all[k]
    preds[k] = pd.Series(model.predict(X_te), index=X_te.index)
    if (k - TRAIN_MIN) % 20 == 0:
        print(f"  [{k}/{len(rebal)}] 训练行数 {len(X_tr)}", flush=True)

# ============ ML 预测 IC（样本外） ============
print("\n" + "=" * 90)
print("LightGBM 预测 IC（样本外，walk-forward）")
print("=" * 90)
ics = [spearman(preds[k], y_all[k]) for k in preds]
s = pd.Series(ics).dropna()
icir = s.mean() / s.std() if s.std() > 0 else 0
print(f"ML 预测 IC = {s.mean():+.4f}  |t| = {tstat(ics):.2f}  ICIR = {icir:.2f}  "
      f"正IC占比 {(s > 0).mean()*100:.0f}%")

# 线性基准 PB+EP 的同期 IC（同预测期对比）
def score_pb_ep(i):
    return -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
ics_lin = [spearman(score_pb_ep(rebal[k]), y_all[k]) for k in preds]
sl = pd.Series(ics_lin).dropna()
print(f"线性 PB+EP 同期 IC = {sl.mean():+.4f}  |t| = {tstat(ics_lin):.2f}")

# ============ 回测 ============
print("\n" + "=" * 90)
print("策略回测（top50 月度含成本）：ML vs 线性 PB+EP")
print("=" * 90)


def universe_ex_small(i, q=SMALL_PCT):
    mi = mcap.iloc[i]
    thr = mi.quantile(q)
    return mi[mi > thr].index


def backtest_score(get_score, universe_fn=None):
    rets, bench, turns = [], [], []
    prev = set()
    for k in range(TRAIN_MIN, len(rebal)):
        i = rebal[k]
        sc = get_score(k, i)
        fwd = fwd_ret(i)
        valid = sc.dropna().index
        if universe_fn is not None:
            valid = valid.intersection(universe_fn(i))
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


def get_ml(k, i):
    return preds[k]


def get_lin(k, i):
    return score_pb_ep(i)


print(f"{'策略':<22}{'年化':>9}{'超额':>8}{'夏普':>7}{'回撤':>8}{'换手':>7}")
print("-" * 90)
bt = {}
for label, gs, uf in [("ML(全样本)", get_ml, None), ("ML(剔小30%)", get_ml, universe_ex_small),
                      ("线性PB+EP(全样本)", get_lin, None), ("线性PB+EP(剔小30%)", get_lin, universe_ex_small)]:
    r = backtest_score(gs, uf)
    bt[label] = r
    print(f"{label:<22}{r[0]:>+8.2f}%{r[1]:>+7.2f}{r[2]:>7.2f}{r[3]:>7.2f}%{r[4]:>6.1f}%")

# ============ 图 ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (1) ML 预测 IC 时序
axes[0].plot(list(preds.keys()), ics, marker='o', markersize=3, linewidth=1, color='crimson', label='ML预测IC')
axes[0].plot(list(preds.keys()), ics_lin, marker='o', markersize=3, linewidth=1, color='steelblue', label='线性PB+EP IC')
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_title("样本外 IC 时序：ML vs 线性")
axes[0].set_xlabel("换仓期")
axes[0].set_ylabel("IC")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# (2) 净值曲线
for label in ["ML(全样本)", "ML(剔小30%)", "线性PB+EP(全样本)", "线性PB+EP(剔小30%)"]:
    ls = '--' if '剔小30%' in label else '-'
    axes[1].plot(bt[label][5].values, linewidth=2, linestyle=ls, label=label)
axes[1].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1].set_title("累计净值（含成本）")
axes[1].set_xlabel("换仓期")
axes[1].set_ylabel("累计净值")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# (3) 特征重要性（最后一次模型，近似）
try:
    lo = max(0, len(rebal) - ROLL_WIN)
    X_tr = pd.concat(X_all[lo:len(rebal)])
    y_tr = pd.concat(y_all[lo:len(rebal)])
    m = y_tr.notna()
    model_last = lgb.LGBMRegressor(**PARAMS)
    model_last.fit(X_tr[m], y_tr[m])
    imp = pd.Series(model_last.feature_importances_, index=FEATURES).sort_values()
    axes[2].barh(imp.index, imp.values, color='seagreen')
    axes[2].set_title("LightGBM 特征重要性（末窗口）")
    axes[2].set_xlabel("重要性")
    axes[2].grid(alpha=0.3, axis='x')
except Exception as e:
    axes[2].text(0.5, 0.5, f"特征重要性计算失败: {e}", ha='center', va='center')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

plt.tight_layout()
plt.savefig("figures/ml_cross_section.png", dpi=150)
print("\n[OK] 图表已保存 figures/ml_cross_section.png")
