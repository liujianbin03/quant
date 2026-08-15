# -*- coding: utf-8 -*-
"""
退市风险过滤：面值过滤（股价 < 阈值剔除）能否避开飞刀、救回反转策略 alpha
- 诊断：踩雷股票在买入时的股价分布
- 测试：真实占比(~5%)下，不同股价地板对策略的影响
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
DELISTED_CACHE = "delisted_cache.pkl"
LOOKBACK, REV, HOLD, N_HOLD = 126, 21, 21, 50
ROUND_TRIP = 0.00025 + 0.001 + 0.00025 + 0.0005 + 0.001


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def run(close_df, turn_df, delist_loss=-0.9, price_floor=None):
    daily_ret = close_df.pct_change(fill_method=None)
    dates = close_df.index
    rets, picks, excluded_delist = [], 0, 0
    prev_held = set()
    i = LOOKBACK
    while i + HOLD < len(dates):
        t0, t1, t_rev, t2 = dates[i-LOOKBACK], dates[i], dates[i-REV], dates[i+HOLD]
        rev = close_df.loc[t1] / close_df.loc[t_rev] - 1
        vol = daily_ret.loc[t0:t1].std()
        turn = turn_df.loc[t_rev:t1].mean()
        fwd = close_df.loc[t2] / close_df.loc[t1] - 1
        dmask = close_df.loc[t1].notna() & close_df.loc[t2].isna()
        fwd = fwd.copy()
        fwd[dmask] = delist_loss

        valid = rev.dropna().index.intersection(vol.dropna().index)\
                            .intersection(turn.dropna().index)
        if price_floor is not None:
            price = close_df.loc[t1]
            # 统计被过滤掉的、即将退市的股票（这些本会踩雷）
            excluded_delist += int((dmask[valid] & (price[valid] < price_floor)).sum())
            valid = valid[price[valid] >= price_floor]

        if len(valid) >= N_HOLD:
            score = -zscore(rev[valid]) - zscore(vol[valid]) - zscore(turn[valid])
            top = score.nlargest(N_HOLD).index
            gross = fwd[top].mean()
            picks += int(dmask[top].sum())
            turnover = 1 - len(set(top) & prev_held) / N_HOLD if prev_held else 1.0
            rets.append(gross - turnover * ROUND_TRIP)
            prev_held = set(top)
        i += HOLD
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    return (cum.iloc[-1] - 1) * 100, (cum.iloc[-1] ** (12 / len(s)) - 1) * 100, \
        picks, excluded_delist


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)
close_l, turn_l = listed['close'], listed['turn']

# ============ 诊断：踩雷股票买入时的股价分布 ============
print("=" * 70)
print("诊断：全部 254 只退市股被策略买中时的股价分布（无过滤）")
print("=" * 70)
close_all = pd.concat([close_l, delisted['close']], axis=1).reindex(close_l.index)
turn_all = pd.concat([turn_l, delisted['turn']], axis=1).reindex(turn_l.index)
daily_ret = close_all.pct_change(fill_method=None)
dates = close_all.index
pick_prices = []
i = LOOKBACK
prev_held = set()
while i + HOLD < len(dates):
    t0, t1, t_rev, t2 = dates[i-LOOKBACK], dates[i], dates[i-REV], dates[i+HOLD]
    rev = close_all.loc[t1] / close_all.loc[t_rev] - 1
    vol = daily_ret.loc[t0:t1].std()
    turn = turn_all.loc[t_rev:t1].mean()
    dmask = close_all.loc[t1].notna() & close_all.loc[t2].isna()
    valid = rev.dropna().index.intersection(vol.dropna().index).intersection(turn.dropna().index)
    if len(valid) >= N_HOLD:
        score = -zscore(rev[valid]) - zscore(vol[valid]) - zscore(turn[valid])
        top = score.nlargest(N_HOLD).index
        for c in top:
            if dmask.get(c, False):
                pick_prices.append(close_all.loc[t1, c])
    i += HOLD
pick_prices = pd.Series(pick_prices, dtype=float)
print(f"踩雷总次数：{len(pick_prices)}")
if len(pick_prices):
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 100)]:
        n = int(((pick_prices >= lo) & (pick_prices < hi)).sum())
        if n:
            print(f"  股价 [{lo},{hi}) 元：{n} 次 ({n/len(pick_prices)*100:.0f}%)")
    print(f"  中位数股价：{pick_prices.median():.2f} 元")
print()

# ============ 测试：真实占比(~5%)下，不同股价地板 ============
print("=" * 70)
print("真实占比(~5%)下，股价过滤效果（退市损失按 -90% 计）")
print("=" * 70)
del_codes = list(delisted['close'].columns)[::8]  # 每8取1
cd = delisted['close'][del_codes]
td = delisted['turn'][del_codes]
close_s = pd.concat([close_l, cd], axis=1).reindex(close_l.index)
turn_s = pd.concat([turn_l, td], axis=1).reindex(turn_l.index)

print(f"{'过滤规则':<22}{'累计收益':>10}{'年化收益':>10}{'踩雷':>6}{'过滤掉退市股':>12}")
print("-" * 70)
for floor, label in [(None, "无过滤"), (1.0, "剔除股价<1元"), (2.0, "剔除股价<2元"),
                     (3.0, "剔除股价<3元"), (5.0, "剔除股价<5元")]:
    total, annual, picks, excl = run(close_s, turn_s, -0.9, floor)
    print(f"{label:<22}{total:>+9.2f}%{annual:>+9.2f}%{picks:>6}{excl:>12}")
print("-" * 70)
print("参考：无退市股(有偏差) 年化 +7.64%；无过滤含退市股 年化 +2.85%")
