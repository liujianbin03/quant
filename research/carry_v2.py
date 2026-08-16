# -*- coding: utf-8 -*-
"""
carry(股息率) 精确版：修 point-in-time 前视 + 样本外验证
- 前视修复：年报4月底前公布(4个月滞后)，故 5月~12月 用上一年度股息，1月~4月 用再上一年度
- 样本外：2021-01 切分，看 carry 及 value+carry 组合是否在样本外仍提升夏普
若样本外仍成立，则把股息率正式加入策略。
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
DIV_CACHE = "dividend_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126
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


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)
with open(DIV_CACHE, 'rb') as f:
    dps = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
daily_ret = close.pct_change(fill_method=None)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]
split_pos = close.index.get_indexer([pd.Timestamp(SPLIT)], method='nearest')[0]


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


def score_value(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    return pbep + t + lc


def known_fiscal_year(dt):
    """年报4月底前公布 → 5月起用上一年度股息，1-4月用再上一年度（无前视）"""
    return dt.year - 1 if dt.month >= 5 else dt.year - 2


def score_carry(i):
    y = known_fiscal_year(dates[i])
    if y < int(dps.index.min()):
        return pd.Series(np.nan, index=close.columns)
    row = dps.loc[y] if y in dps.index else pd.Series(np.nan, index=close.columns)
    row = row.reindex(close.columns)
    return zscore(winsorize(row / close.iloc[i]))


def backtest(score_fn, rebal_list):
    rets, prev, ret_dates = [], set(), []
    for i in rebal_list:
        sc = score_fn(i)
        fwd = close.iloc[i + HOLD] / close.iloc[i] - 1
        v = sc.dropna().index.intersection(universe_ex_small(i))
        if len(v) >= N_HOLD:
            top = sc[v].nlargest(N_HOLD).index
            turnv = 1 - len(set(top) & prev) / N_HOLD if prev else 1.0
            rets.append(fwd[top].mean() - turnv * ROUND_TRIP)
            prev = set(top)
            ret_dates.append(dates[i])
    return pd.Series(rets, index=pd.Index(ret_dates))


def stats(s):
    if len(s) < 10:
        return 0, 0, 0, 0
    cum = (1 + s).cumprod()
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    return annual, sharpe, mdd, cum


in_rebal = [i for i in rebal if i < split_pos]
out_rebal = [i for i in rebal if i >= split_pos]

print("=" * 92)
print("carry 精确版（point-in-time 5月滞后）+ 样本外验证")
print("=" * 92)

# 权重扫描：股息率权重 w ∈ {0, 0.2, 0.3, 0.5, 1.0}
print(f"{'股息率权重':<10}{'全样本夏普':>10}{'全样本年化':>12}{'样本外夏普':>10}{'样本外年化':>12}")
print("-" * 92)
for w in [0, 0.2, 0.3, 0.5, 1.0]:
    fn = lambda i, w=w: score_value(i) + w * score_carry(i).fillna(0)
    s_full = backtest(fn, rebal)
    s_out = backtest(fn, out_rebal)
    a_f, sh_f, dd_f, _ = stats(s_full)
    a_o, sh_o, dd_o, _ = stats(s_out)
    print(f"{w:<10}{sh_f:>+10.2f}{a_f:>+11.2f}%{sh_o:>+10.2f}{a_o:>+11.2f}%")

print("-" * 92)
print("判断：若加股息率后样本外夏普 > 纯价值(权重0)样本外夏普，则股息率是稳健增量，可采纳。")
