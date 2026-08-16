# -*- coding: utf-8 -*-
"""
纸面跟踪（paper trade）：零风险地前向验证 PB+EP+换手率+涨停次数 + 剔小30% 策略
用法（每次刷新数据后跑一次，建议月度）：
    python paper_trade.py

机制：
  - 首次运行：记录当前 top20 持仓 + 起始净值=1.0 + 沪深300 起始点位
  - 之后每次运行：计算上次持仓至今的前向收益（含换手成本），滚动累积净值，
    与沪深300 同期对比，并记录到 paper_trade_history.csv
  - 每次都会打印「当前持仓 + 累计净值 vs 沪深300 + 换手」

这是比回测更诚实的验证：回测有幸存者偏差/参数泄漏，纸面跟踪是零风险的实盘预演。
"""
import os
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
INDEX_NAME_CACHE = "index_name_cache.pkl"
DIV_CACHE = "dividend_cache.pkl"

HOLD = 21
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
N_HOLD = 20
STATE = "paper_trade_state.pkl"
HISTORY = "paper_trade_history.csv"


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


# ============ 数据 ============
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
names = iname['name'].reindex(close.columns)
hs300 = iname['index'].reindex(close.index)
with open(DIV_CACHE, 'rb') as f:
    dps = pickle.load(f)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
daily_ret = close.pct_change(fill_method=None)
dates = close.index


def carry_yield(i):
    """股息率：上一财年派息(税前)/现价，4个月年报滞后(5月起用上年、1-4月用再上年)"""
    y = dates[i].year - 1 if dates[i].month >= 5 else dates[i].year - 2
    if y < int(dps.index.min()):
        return pd.Series(np.nan, index=close.columns)
    row = dps.loc[y] if y in dps.index else pd.Series(np.nan, index=close.columns)
    return row.reindex(close.columns) / close.iloc[i]


def score_strategy(i):
    """PB+EP+股息率 + 换手率(低) + 涨停次数(少) —— 参数搜索+carry样本外验证"""
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    cy = zscore(winsorize(carry_yield(i))).fillna(0)
    return pbep + t + lc + 0.3 * cy


def universe_ex_small(i):
    mi = mcap.iloc[i]
    thr = mi.quantile(SMALL_PCT)
    return mi[mi > thr].index


def current_top(i):
    sc = score_strategy(i)
    valid = sc.dropna().index.intersection(universe_ex_small(i))
    return sc[valid].nlargest(N_HOLD)


i = len(dates) - 1
top = current_top(i)
now = dates[i]
idx_now = hs300.iloc[i]

# ============ 状态机 ============
if os.path.exists(STATE):
    with open(STATE, 'rb') as f:
        st = pickle.load(f)

    # 上次持仓的前向收益
    old_hold = st['holdings']  # {code: close_then}
    held = [c for c in old_hold if pd.notna(close.iloc[i].get(c))]
    fwd = float(np.mean([close.iloc[i][c] / old_hold[c] - 1 for c in held])) if held else 0.0
    overlap = len(set(top.index) & set(old_hold)) / N_HOLD
    cost = (1 - overlap) * ROUND_TRIP
    ret = fwd - cost

    # 累计净值
    st['nav'] = st['nav'] * (1 + ret)
    idx_ret = idx_now / st['index'] - 1
    st['index'] = idx_now
    st['holdings'] = {c: float(close.iloc[i][c]) for c in top.index}
    days = (now - st['date']).days
    st['date'] = now

    # 记录历史
    rec = {"date": str(now.date()), "period_return": round(ret * 100, 2),
           "nav": round(st['nav'], 4), "hs300_return": round(idx_ret * 100, 2),
           "turnover": round((1 - overlap) * 100, 1)}
    hist = pd.DataFrame([rec])
    hist.to_csv(HISTORY, mode='a', header=not os.path.exists(HISTORY), index=False)

    print("=" * 78)
    print(f"纸面跟踪更新  本次调仓日 {now.date()}（距上次 {days} 天）")
    print("=" * 78)
    print(f"  本期策略收益(含换手成本): {ret*100:+.2f}%  (换手 {((1-overlap)*100):.0f}%)")
    print(f"  本期沪深300:               {idx_ret*100:+.2f}%")
    print(f"  累计净值: {st['nav']:.4f}  |  沪深300累计: {idx_now / st['nav0_index']:.4f}  |  "
          f"相对超额: {(st['nav'] - idx_now / st['nav0_index']) * 100:+.2f}%")
else:
    st = {'date': now, 'nav': 1.0, 'index': idx_now, 'nav0_index': idx_now,
          'holdings': {c: float(close.iloc[i][c]) for c in top.index}}
    print("=" * 78)
    print(f"纸面跟踪已启动  起始日 {now.date()}  起始净值 1.0")
    print("=" * 78)
    print("下次运行本脚本将计算前向收益并累积净值。建议每月跑一次。")

with open(STATE, 'wb') as f:
    pickle.dump(st, f)

# ============ 当前持仓 ============
print("\n当前持仓 top%d:" % N_HOLD)
print(f"{'排名':<5}{'代码':<12}{'名称':<10}{'现价':>8}")
print("-" * 40)
for rank, code in enumerate(top.index, 1):
    nm = names[code] if pd.notna(names[code]) else '?'
    print(f"{rank:<5}{code:<12}{nm:<10}{close.iloc[i][code]:>8.2f}")

# 起始至今累计（若有）
if 'nav0_index' in st and st['nav'] != 1.0:
    total_hs = idx_now / st['nav0_index'] - 1
    total_strat = st['nav'] - 1
    print(f"\n自 {st['date'].date()} 至今：策略 {total_strat*100:+.2f}%  vs  沪深300 {total_hs*100:+.2f}%")
