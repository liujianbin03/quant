# -*- coding: utf-8 -*-
"""
EP / PB / PB+EP 的鲁棒性检验：把之前两脚本的"样本内、幸存者偏差未修正"结论补硬

Part 1 — 幸存者偏差修正（退市股）：
    把 2015 后 254 只退市股按不同占比加回股票池，持有期退市 fwd=-90%，
    看 PB / EP / PB+EP 三者的 alpha 各自被侵蚀多少、踩雷多少次。
    依据 README：反转策略年化被幸存者偏差虚增 ~4.8pp，而 PB 仅 -0.28pp。
    本脚本验证：EP 是否和 PB 一样天然规避退市雷区。

Part 2 — 样本外验证（2021-01-04 切分）：
    IC 方向 + 策略年化，样本内(2015-2021) vs 样本外(2021-2026)，
    全样本 与 剔小30% 两种口径，判定"EP / PB+EP 剔小30%后反超"是 alpha 还是噪声。

口径：分数高=便宜；月度调仓 top50 等权；ROUND_TRIP=0.003；区间 2015-2026。
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

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.0030
DELIST_LOSS = -0.9      # 持有期退市 → fwd 记 -90%（项目既定口径）
SMALL_PCT = 0.30
SPLIT = "2021-01-04"


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
    """闭包：因子分数（高=便宜）。EP 全样本口径（含亏损，亏损股自然垫底）"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ep_df = (1.0 / pe_df).replace([np.inf, -np.inf], np.nan)

    def score_pb(i):
        return -zscore(clean(pb_df.iloc[i]))

    def score_ep(i):
        return zscore(winsorize(ep_df.iloc[i]))

    def score_pb_ep(i):
        return score_pb(i) + score_ep(i)

    return {"PB": score_pb, "EP": score_ep, "PB+EP": score_pb_ep}, ep_df


def backtest(close_df, score_fn, rebal, delist_loss=None, universe_fn=None):
    """通用回测：支持退市损失 + 自定义股票池过滤"""
    rets, turns, picks = [], [], 0
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
        if universe_fn is not None:
            valid = valid.intersection(universe_fn(i))
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
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100 if len(s) else 0.0
    total = (cum.iloc[-1] - 1) * 100 if len(s) else 0.0
    sharpe = s.mean() / s.std() * np.sqrt(12) if len(s) and s.std() > 0 else 0.0
    mdd = (cum / cum.cummax() - 1).min() * 100 if len(s) else 0.0
    return dict(annual=annual, total=total, sharpe=sharpe, mdd=mdd,
                turn=np.mean(turns) * 100 if turns else 0.0, picks=picks, cum=cum)


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

close_l = listed['close']
pe_l = val['pe'].reindex(close_l.index)
pb_l = val['pb'].reindex(close_l.index)
total_share = size['totalShare'].reindex(close_l.columns)

del_codes = list(delisted['close'].columns)
close_d = delisted['close']
pe_d = dval['pe'].reindex(close_d.index)
pb_d = dval['pb'].reindex(close_d.index)

dates = close_l.index
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates)]
split_pos = close_l.index.get_indexer([pd.Timestamp(SPLIT)], method='nearest')[0]
in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]

mcap = close_l.mul(total_share, axis=1)


def universe_ex_small(i, q=SMALL_PCT):
    mi = mcap.iloc[i]
    thr = mi.quantile(q)
    return mi[mi > thr].index


# ============ Part 1：幸存者偏差修正 ============
print("=" * 96)
print("Part 1  幸存者偏差修正：加入退市股（持有期退市 fwd=-90%）")
print("=" * 96)
print(f"{'退市股抽样':<16}{'退市占比':>9}{'PB年化':>9}{'EP年化':>9}{'PB+EP年化':>11}"
      f"{'PB踩雷':>8}{'EP踩雷':>8}{'PBE踩雷':>8}")
print("-" * 96)

# 基线（无退市股）
scores_l, _ = make_scores(pe_l, pb_l)
base = {n: backtest(close_l, scores_l[n], rebal) for n in scores_l}
print(f"{'无退市(基线)':<16}{0:>8.1f}%{base['PB']['annual']:>+9.2f}%{base['EP']['annual']:>+9.2f}%"
      f"{base['PB+EP']['annual']:>+11.2f}%{base['PB']['picks']:>8}{base['EP']['picks']:>8}"
      f"{base['PB+EP']['picks']:>8}")

surv = {}   # (sampling_label) -> {factor: annual}
picks_s = {}
for step, label in [(1, "全部254只"), (4, "每4取1(~64)"), (8, "每8取1(~32)")]:
    sample = del_codes[::step]
    cd = close_d[sample].reindex(close_l.index)
    pe_dd = pe_d[sample].reindex(close_l.index)
    pb_dd = pb_d[sample].reindex(close_l.index)

    close_all = pd.concat([close_l, cd], axis=1)
    pe_all = pd.concat([pe_l, pe_dd], axis=1)
    pb_all = pd.concat([pb_l, pb_dd], axis=1)

    scores_a, _ = make_scores(pe_all, pb_all)
    ratio = len(sample) / (close_l.shape[1] + len(sample)) * 100
    row = {}
    pk = {}
    for n in scores_a:
        r = backtest(close_all, scores_a[n], rebal, delist_loss=DELIST_LOSS)
        row[n] = r['annual']
        pk[n] = r['picks']
    surv[label] = row
    picks_s[label] = pk
    print(f"{label:<16}{ratio:>8.1f}%{row['PB']:>+9.2f}%{row['EP']:>+9.2f}%{row['PB+EP']:>+11.2f}%"
          f"{pk['PB']:>8}{pk['EP']:>8}{pk['PB+EP']:>8}")

print("-" * 96)
print("侵蚀幅度（基线年化 - 含退市年化，越小越抗幸存者偏差）：")
for label in surv:
    line = f"  {label:<14}"
    for n in ["PB", "EP", "PB+EP"]:
        line += f" {n} {base[n]['annual']-surv[label][n]:+.2f}pp"
    print(line)

# ============ Part 2：样本外验证 ============
print("\n" + "=" * 96)
print(f"Part 2  样本外验证（切分 {dates[split_pos].date()}）")
print("=" * 96)


def ic_split(score_fn, ep_series=None):
    """返回 (样本内IC, |t|, 样本外IC, |t|)"""
    ics_in = [spearman(score_fn(i), close_l.iloc[i + HOLD] / close_l.iloc[i] - 1) for i in in_rebal]
    ics_out = [spearman(score_fn(i), close_l.iloc[i + HOLD] / close_l.iloc[i] - 1) for i in out_rebal]
    return (pd.Series(ics_in).dropna().mean(), tstat(ics_in),
            pd.Series(ics_out).dropna().mean(), tstat(ics_out))


print(f"{'因子':<10}{'口径':<8}{'样本内IC':>10}{'|t|':>7}{'样本外IC':>10}{'|t|':>7}{'方向稳?':>8}")
print("-" * 96)

for n, fn in scores_l.items():
    m_in, t_in, m_out, t_out = ic_split(fn)
    stable = "是" if m_in * m_out > 0 and abs(t_out) >= 2 else "否"
    print(f"{n:<10}{'全样本':<8}{m_in:>+10.4f}{t_in:>7.2f}{m_out:>+10.4f}{t_out:>7.2f}{stable:>8}")

    # 剔小30% 口径
    ics_in_x = [spearman(fn(i)[universe_ex_small(i)],
                         (close_l.iloc[i + HOLD] / close_l.iloc[i] - 1)[universe_ex_small(i)]) for i in in_rebal]
    ics_out_x = [spearman(fn(i)[universe_ex_small(i)],
                          (close_l.iloc[i + HOLD] / close_l.iloc[i] - 1)[universe_ex_small(i)]) for i in out_rebal]
    m_in2, t_in2 = pd.Series(ics_in_x).dropna().mean(), tstat(ics_in_x)
    m_out2, t_out2 = pd.Series(ics_out_x).dropna().mean(), tstat(ics_out_x)
    stable2 = "是" if m_in2 * m_out2 > 0 and abs(t_out2) >= 2 else "否"
    print(f"{'':<10}{'剔小30%':<8}{m_in2:>+10.4f}{t_in2:>7.2f}{m_out2:>+10.4f}{t_out2:>7.2f}{stable2:>8}")
    print("-" * 96)

# 策略分期间回测
print(f"{'因子':<10}{'口径':<8}{'样本内年化':>12}{'样本外年化':>12}{'样本外夏普':>10}{'样本外回撤':>10}")
print("-" * 96)
oos_nav = {}
for n, fn in scores_l.items():
    r_full_in = backtest(close_l, fn, in_rebal)
    r_full_out = backtest(close_l, fn, out_rebal)
    r_ex_in = backtest(close_l, fn, in_rebal, universe_fn=universe_ex_small)
    r_ex_out = backtest(close_l, fn, out_rebal, universe_fn=universe_ex_small)
    oos_nav[n] = (r_full_out['cum'], r_ex_out['cum'])
    print(f"{n:<10}{'全样本':<8}{r_full_in['annual']:>+12.2f}{r_full_out['annual']:>+12.2f}"
          f"{r_full_out['sharpe']:>10.2f}{r_full_out['mdd']:>10.2f}%")
    print(f"{'':<10}{'剔小30%':<8}{r_ex_in['annual']:>+12.2f}{r_ex_out['annual']:>+12.2f}"
          f"{r_ex_out['sharpe']:>10.2f}{r_ex_out['mdd']:>10.2f}%")
    print("-" * 96)

# ============ 图 ============
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# (1) 幸存者偏差：年化侵蚀
labels = list(surv.keys())
x = np.arange(len(labels))
w = 0.26
for j, n in enumerate(["PB", "EP", "PB+EP"]):
    erosion = [base[n]['annual'] - surv[l][n] for l in labels]
    axes[0, 0].bar(x + (j - 1) * w, erosion, w, label=n)
axes[0, 0].axhline(0, color='black', linewidth=0.8)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].set_ylabel("年化侵蚀 (pp)")
axes[0, 0].set_title("幸存者偏差侵蚀：加入退市股后年化下降幅度（越小越抗）")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3, axis='y')

# (2) 样本外 IC
names = list(scores_l.keys())
ins = [pd.Series([spearman(scores_l[n](i), close_l.iloc[i+HOLD]/close_l.iloc[i]-1) for i in in_rebal]).dropna().mean() for n in names]
outs = [pd.Series([spearman(scores_l[n](i), close_l.iloc[i+HOLD]/close_l.iloc[i]-1) for i in out_rebal]).dropna().mean() for n in names]
x2 = np.arange(len(names))
axes[0, 1].bar(x2 - 0.2, ins, 0.4, label='样本内(2015-2021)', color='steelblue')
axes[0, 1].bar(x2 + 0.2, outs, 0.4, label='样本外(2021-2026)', color='orange')
axes[0, 1].axhline(0, color='black', linewidth=0.8)
axes[0, 1].set_xticks(x2)
axes[0, 1].set_xticklabels(names)
axes[0, 1].set_title("IC：样本内 vs 样本外（全样本口径）")
axes[0, 1].set_ylabel("IC 均值")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3, axis='y')

# (3) 样本外净值
for n in names:
    axes[1, 0].plot(oos_nav[n][0].values, linewidth=2, label=f"{n}(全样本)")
    axes[1, 0].plot(oos_nav[n][1].values, linewidth=2, linestyle='--', label=f"{n}(剔小30%)")
axes[1, 0].axhline(1.0, color='black', linewidth=0.5, alpha=0.5)
axes[1, 0].set_title("样本外(2021-2026)累计净值")
axes[1, 0].set_xlabel("换仓期")
axes[1, 0].set_ylabel("累计净值")
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(alpha=0.3)

# (4) 幸存者偏差：踩雷数
for j, n in enumerate(["PB", "EP", "PB+EP"]):
    cnts = [picks_s[l][n] for l in labels]
    axes[1, 1].bar(x + (j - 1) * w, cnts, w, label=n)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_ylabel("踩雷次数（选到退市股）")
axes[1, 1].set_title("踩雷：top50 选中退市股的次数")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figures/value_ep_robustness.png", dpi=150)
print("\n[OK] 图表已保存 figures/value_ep_robustness.png")
