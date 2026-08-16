# -*- coding: utf-8 -*-
"""
绩效与归因报告（目标要求的"每两周输出一次"交付物）
1) 绩效：当前策略(PB+EP+换手率+涨停次数, top20, 剔小30%) 的累计净值 vs 沪深300 vs 等权基准，
        年化/夏普/最大回撤/近12期滚动。
2) 归因：持仓相对全市场（剔小30%候选池）的因子倾斜 —— 说明策略"赌的是什么"。
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
INDEX_NAME_CACHE = "index_name_cache.pkl"
DIV_CACHE = "dividend_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126


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


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)
with open(INDEX_NAME_CACHE, 'rb') as f:
    iname = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
hs300 = iname['index'].reindex(close.index)
with open(DIV_CACHE, 'rb') as f:
    dps = pickle.load(f)
daily_ret = close.pct_change(fill_method=None)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]


def carry_yield(i):
    """股息率：上一财年派息(税前)/现价，4个月年报滞后(5月起用上年、1-4月用再上年)"""
    y = dates[i].year - 1 if dates[i].month >= 5 else dates[i].year - 2
    if y < int(dps.index.min()):
        return pd.Series(np.nan, index=close.columns)
    row = dps.loc[y] if y in dps.index else pd.Series(np.nan, index=close.columns)
    return row.reindex(close.columns) / close.iloc[i]


def score_strategy(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    cy = zscore(winsorize(carry_yield(i))).fillna(0)
    return pbep + t + lc + 0.3 * cy


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


# ============ 绩效 ============
rets, bench, hs_ret = [], [], []
prev = set()
for i in rebal:
    sc = score_strategy(i)
    fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
    v = sc.dropna().index.intersection(universe_ex_small(i))
    bench.append(fwd[v].mean())
    hs_ret.append(hs300.iloc[i + HOLD] / hs300.iloc[i] - 1)
    if len(v) >= N_HOLD:
        top = sc[v].nlargest(N_HOLD).index
        turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
        rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
        prev = set(top)

s = pd.Series(rets)
b = pd.Series(bench)
h = pd.Series(hs_ret)
cum = (1 + s).cumprod()
b_cum = (1 + b).cumprod()
h_cum = (1 + h).cumprod()
annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
sharpe = s.mean() / s.std() * np.sqrt(12)
mdd = (cum / cum.cummax() - 1).min() * 100

print("=" * 76)
print("绩效与归因报告  （策略：PB+EP+换手率+涨停次数，top20，剔小30%，含成本）")
print("=" * 76)
print("【1. 绩效】")
print(f"  累计净值    : 策略 {cum.iloc[-1]:.2f}  |  等权基准 {b_cum.iloc[-1]:.2f}  |  沪深300 {h_cum.iloc[-1]:.2f}")
print(f"  年化收益    : {annual:+.2f}%")
print(f"  夏普        : {sharpe:.2f}")
print(f"  最大回撤    : {mdd:.2f}%")
print(f"  近12期(≈1年) : 策略 {(1+s[-12:]).prod()-1:+.2%}  |  沪深300 {(1+h[-12:]).prod()-1:+.2%}")

# ============ 归因 ============
i = len(dates) - 1
sc = score_strategy(i)
v = sc.dropna().index.intersection(universe_ex_small(i))
top = sc[v].nlargest(N_HOLD)
uni = v

print("\n【2. 归因：当前持仓 vs 候选池的因子倾斜】")
print(f"{'因子':<14}{'持仓均值':>10}{'候选池均值':>12}{'倾斜方向':>16}")
print("-" * 76)
rows = [
    ("PB(低好)", pb.iloc[i][top.index].median(), pb.iloc[i][uni].median(), "低估"),
    ("EP%(高好)", ep.iloc[i][top.index].median() * 100, ep.iloc[i][uni].median() * 100, "高盈利"),
    ("换手率(低好)", turn.iloc[i - 21:i].mean()[top.index].median(), turn.iloc[i - 21:i].mean()[uni].median(), "低关注"),
    ("涨停次数(少好)", (daily_ret.iloc[i - 63:i] >= 0.095).sum()[top.index].mean(), (daily_ret.iloc[i - 63:i] >= 0.095).sum()[uni].mean(), "少投机"),
]
for name, a, bv, tilt in rows:
    print(f"{name:<14}{a:>10.2f}{bv:>12.2f}{tilt:>16}")

print("\n【3. 与目标对标（修订后）】")
print(f"  目标: 跑赢沪深300(超额≥0) / 最大回撤≤15% / 夏普≥0.4")
excess_b = (cum.iloc[-1] - b_cum.iloc[-1]) * 100
excess_h = (cum.iloc[-1] - h_cum.iloc[-1]) * 100
print(f"  超额: 相对等权 {excess_b:+.1f}pp  {'✓' if excess_b>=0 else '✗'}  |  相对沪深300 {excess_h:+.1f}pp  {'✓' if excess_h>=0 else '✗'}")
print(f"  回撤: {mdd:.1f}%  {'✓≤15%' if mdd>=-15 else '✗>15%'}  |  夏普: {sharpe:.2f}  {'✓≥0.4' if sharpe>=0.4 else '✗<0.4'}")

# 纸面跟踪实盘状态
import os
if os.path.exists("paper_trade_state.pkl"):
    with open("paper_trade_state.pkl", 'rb') as f:
        st = pickle.load(f)
    print("\n【4. 纸面跟踪实盘状态】")
    print(f"  起始 {st['date'].date()}，当前净值 {st['nav']:.4f}，沪深300起始 {st.get('nav0_index', st['index']):.0f}")
    if os.path.exists("paper_trade_history.csv"):
        hist_df = pd.read_csv("paper_trade_history.csv")
        print(f"  已记录 {len(hist_df)} 期，累计净值 {hist_df['nav'].iloc[-1]:.4f}")
    else:
        print("  尚无历史记录（等待下次月度数据刷新后运行 paper_trade.py）")

# ============ 持久化报告（Markdown） ============
import datetime
os.makedirs("reports", exist_ok=True)
report_date = datetime.date.today().isoformat()
md = []
md.append(f"# 绩效与归因报告  {report_date}\n")
md.append("策略：PB+EP+换手率+涨停次数，top20，剔小30%，月度含成本\n")
md.append("## 绩效\n")
md.append(f"- 累计净值：策略 {cum.iloc[-1]:.2f} | 等权 {b_cum.iloc[-1]:.2f} | 沪深300 {h_cum.iloc[-1]:.2f}")
md.append(f"- 年化：{annual:+.2f}%")
md.append(f"- 夏普：{sharpe:.2f}")
md.append(f"- 最大回撤：{mdd:.2f}%")
md.append(f"- 近12期(≈1年)：策略 {(1+s[-12:]).prod()-1:+.2%} | 沪深300 {(1+h[-12:]).prod()-1:+.2%}\n")
md.append("## 归因（当前持仓 vs 候选池）\n")
md.append("| 因子 | 持仓均值 | 候选池均值 | 倾斜 |")
md.append("|---|---|---|---|")
for name, a, bv, tilt in rows:
    md.append(f"| {name} | {a:.2f} | {bv:.2f} | {tilt} |")
md.append("\n## 目标对标（修订后）\n")
md.append(f"- 跑赢沪深300(超额≥0)：超额 {excess_h:+.1f}pp {'✓' if excess_h>=0 else '✗'}")
md.append(f"- 最大回撤≤15%：{mdd:.1f}% {'✓' if mdd>=-15 else '✗'}")
md.append(f"- 夏普≥0.4：{sharpe:.2f} {'✓' if sharpe>=0.4 else '✗'}")
path = f"reports/report_{report_date}.md"
with open(path, 'w', encoding='utf-8') as f:
    f.write("\n".join(md))
print(f"\n[报告已保存] {path}")
