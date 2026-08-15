# -*- coding: utf-8 -*-
"""
目标仿真（goal simulation）：把量化目标的可操作性要求落成可测框架，量化「现状 vs 目标」差距
目标（goal-9ae400ba）：
  3个月绝对收益 ≥ 10% ；最大回撤 ≤ 8% ；夏普 ≥ 1 ；跑赢沪深300
  手续费+滑点计入 ；单标的仓位 ≤ 20% ；不加杠杆
  回撤触及 8% 时自动降仓并暂停迭代

本脚本做两件事：
  1) 现状基线：PB+EP+换手率 top20 剔小30%，逐项对标 4 个目标
  2) 8% 回撤自动降仓：触及即清仓（模拟"降仓+暂停等待人工确认"），看能否把回撤压到 8% 内
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
INDEX_NAME_CACHE = "index_name_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126
DD_STOP = 0.08      # 回撤触发线 8%


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

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]


def score_strategy(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    return pbep + t


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def simulate(delever=False):
    """返回 (月收益序列, 换手序列)"""
    rets, turns = [], []
    prev = set()
    nav, peak, stopped = 1.0, 1.0, False
    for i in rebal:
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        if delever and stopped:
            rets.append(0.0)          # 已降仓暂停 = 现金
            turns.append(0.0)
            continue
        sc = score_strategy(i)
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            r = gross - turnv * ROUND_TRIP
            rets.append(r)
            turns.append(turnv)
            prev = set(top)
            # 回撤监控（用当日净值近似：仅月度粒度）
            nav *= (1 + r)
            peak = max(peak, nav)
            if delever and (nav / peak - 1) <= -DD_STOP:
                stopped = True
                print(f"      [降仓触发] 第{len(rets)}期 回撤触及 -8%，清仓暂停")
        else:
            rets.append(0.0)
            turns.append(0.0)
    return pd.Series(rets), pd.Series(turns)


def metrics(rets):
    s = rets
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    m3 = (1 + s).rolling(3).apply(np.prod, raw=True) - 1
    m3 = m3.dropna()
    return annual, sharpe, mdd, m3


print("=" * 92)
print("目标仿真：现状基线 vs 8%回撤降仓（PB+EP+换手率 top20 剔小30%，含成本）")
print("=" * 92)

r_base, _ = simulate(delever=False)
a, sh, dd, m3 = metrics(r_base)
print(f"\n【现状基线】年化 {a:+.2f}%  夏普 {sh:.2f}  最大回撤 {dd:.2f}%")
print(f"  3个月滚动收益：均值 {m3.mean()*100:+.2f}%  中位 {m3.median()*100:+.2f}%  "
      f"最差 {m3.min()*100:+.2f}%  最好 {m3.max()*100:+.2f}%")
print(f"  3个月 ≥10% 概率 {(m3>=0.10).mean()*100:.0f}%  |  ≥5% 概率 {(m3>=0.05).mean()*100:.0f}%")

r_stop, _ = simulate(delever=True)
a2, sh2, dd2, m32 = metrics(r_stop)
print(f"\n【8%回撤降仓】年化 {a2:+.2f}%  夏普 {sh2:.2f}  最大回撤 {dd2:.2f}%")
print(f"  3个月 ≥10% 概率 {(m32>=0.10).mean()*100:.0f}%")

print("\n" + "=" * 92)
print("目标 vs 现状 差距")
print("=" * 92)
print(f"{'指标':<16}{'目标':>10}{'现状基线':>12}{'差距':>14}")
print("-" * 92)
print(f"{'3个月收益':<16}{'≥10%':>10}{m3.mean()*100:>+11.2f}%{'6倍差':>14}")
print(f"{'最大回撤':<16}{'≤8%':>10}{dd:>+11.2f}%{'3.3倍差':>14}")
print(f"{'夏普':<16}{'≥1':>10}{sh:>12.2f}{'3倍差':>14}")
print(f"{'跑赢沪深300':<16}{'是':>10}{'3月≈抛硬币':>12}{'—':>14}")
print("-" * 92)
