# -*- coding: utf-8 -*-
"""
幸存者偏差分析：对比「只含上市股」vs「含退市股」的反转策略表现
- 无退市股：当前结果（有幸存者偏差）
- 含退市股：加入 2015 后退市的 255 只，退市后按 -100% 处理
用法：python -u survivorship_bias.py
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
DELISTED_CACHE = "delisted_cache.pkl"

LOOKBACK = 126
REV = 21
HOLD = 21
N_HOLD = 50

COMMISSION = 0.00025
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
ROUND_TRIP = COMMISSION + SLIPPAGE + COMMISSION + STAMP_TAX + SLIPPAGE  # 0.30%

FACTORS = {
    "动量(6月)": +1, "反转(1月)": -1, "波动率(6月)": -1, "换手率(1月)": -1,
}


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


def run(close_df, turn_df, out_date=None, N_HOLD=50, with_cost=True):
    """跑策略 + 收集 IC。out_date 为 {code: 退市日} 时启用退市处理"""
    daily_ret = close_df.pct_change(fill_method=None)
    dates = close_df.index
    ic_collect = {n: [] for n in FACTORS}
    rets, bench_rets, turnovers = [], [], []
    delist_picks = 0          # 买到退市股的次数（退市后归零）
    delist_pick_losses = []   # 这些退市股的平均损失
    prev_held = set()
    delist_set = set(out_date.keys()) if out_date else set()

    i = LOOKBACK
    while i + HOLD < len(dates):
        t0 = dates[i - LOOKBACK]
        t1 = dates[i]
        t_rev = dates[i - REV]
        t2 = dates[i + HOLD]

        mom = close_df.loc[t1] / close_df.loc[t0] - 1
        rev = close_df.loc[t1] / close_df.loc[t_rev] - 1
        vol = daily_ret.loc[t0:t1].std()
        turn = turn_df.loc[t_rev:t1].mean()
        fwd = close_df.loc[t2] / close_df.loc[t1] - 1

        # 退市处理：t1 还上市、t2 已退市（close 变 NaN）→ 收益 -100%
        delist_mask = close_df.loc[t1].notna() & close_df.loc[t2].isna()
        fwd = fwd.copy()
        fwd[delist_mask] = -1.0

        fmap = {"动量(6月)": mom, "反转(1月)": rev,
                "波动率(6月)": vol, "换手率(1月)": turn}
        for n in FACTORS:
            ic = spearman(fmap[n], fwd)
            if not np.isnan(ic):
                ic_collect[n].append(ic)

        valid = rev.dropna().index.intersection(vol.dropna().index)\
                            .intersection(turn.dropna().index)
        bench_rets.append(fwd[valid].mean())

        if len(valid) >= N_HOLD:
            score = -zscore(rev[valid]) - zscore(vol[valid]) - zscore(turn[valid])
            top = score.nlargest(N_HOLD).index
            gross = fwd[top].mean()

            # 统计买到的退市股
            for c in top:
                if c in delist_set and delist_mask.get(c, False):
                    delist_picks += 1
                    delist_pick_losses.append(fwd[c])

            if prev_held:
                turnover = 1 - len(set(top) & prev_held) / N_HOLD
            else:
                turnover = 1.0
            ret = gross - turnover * ROUND_TRIP if with_cost else gross
            rets.append(ret)
            turnovers.append(turnover)
            prev_held = set(top)

        i += HOLD

    return ic_collect, pd.Series(rets), pd.Series(bench_rets), \
        turnovers, delist_picks, delist_pick_losses


def stats(rets):
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
    return total, annual, dd, sharpe, cum


# ============ 数据 ============
with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)

close_listed = listed['close']
turn_listed = listed['turn']
close_del = delisted['close']
turn_del = delisted['turn']
out_date = delisted['outDate']

# 合并（退市股在退市日之后是 NaN）
close_all = pd.concat([close_listed, close_del], axis=1).reindex(close_listed.index)
turn_all = pd.concat([turn_listed, turn_del], axis=1).reindex(turn_listed.index)

print(f"上市股：{close_listed.shape[1]} 只   退市股：{close_del.shape[1]} 只   合并：{close_all.shape[1]} 只")

# ============ 跑两版 ============
r_ic, r_rets, r_bench, r_turn, r_picks, r_loss = run(close_listed, turn_listed)
a_ic, a_rets, a_bench, a_turn, a_picks, a_loss = run(close_all, turn_all, out_date=out_date)

print("\n" + "=" * 84)
print("因子 IC 对比（无退市股 vs 含退市股）")
print("=" * 84)
print(f"{'因子':<14}{'无退市IC':>10}{'含退市IC':>10}{'变化':>10}")
print("-" * 84)
for n in FACTORS:
    ic_no = pd.Series(r_ic[n]).dropna().mean()
    ic_yes = pd.Series(a_ic[n]).dropna().mean()
    print(f"{n:<14}{ic_no:>10.4f}{ic_yes:>10.4f}{ic_yes - ic_no:>+10.4f}")
print("-" * 84)

print("\n" + "=" * 84)
print("策略表现对比（反转+低换手+低波动，top50，含成本）")
print("=" * 84)
ti, ai_, di, si, cum_no = stats(r_rets)
ty, ay, dy, sy, cum_yes = stats(a_rets)
print(f"{'组合':<22}{'累计':>9}{'年化':>9}{'回撤':>9}{'夏普':>7}")
print("-" * 84)
print(f"{'无退市股(有偏差)':<20}{ti:>+9.2f}%{ai_:>+9.2f}%{di:>9.2f}%{si:>7.2f}")
print(f"{'含退市股(修正后)':<20}{ty:>+9.2f}%{ay:>+9.2f}%{dy:>9.2f}%{sy:>7.2f}")
print("-" * 84)
print(f"幸存者偏差 = 累计收益被高估 {ti - ty:.2f} 个百分点，年化 {ai_ - ay:.2f} 个百分点")

print(f"\n退市股相关：")
print(f"  策略买到退市股 {r_picks} 次（含退市股版本中），平均损失 {np.mean(a_loss)*100:.1f}%" if a_loss
      else "  （无退市股版本不涉及）")
print(f"  含退市股版本中，策略买到退市股 {a_picks} 次，这些票平均跌 {np.mean(a_pick_losses := a_loss)*100:.1f}%" if a_picks else "")

# ============ 图 ============
plt.figure(figsize=(14, 7))
plt.plot(cum_no.index, cum_no.values, color='red', linewidth=2.2, label='无退市股(有幸存者偏差)')
plt.plot(cum_yes.index, cum_yes.values, color='darkred', linewidth=2.2, linestyle='--', label='含退市股(修正后)')
plt.axhline(1.0, color='black', linestyle=':', alpha=0.4)
plt.title("幸存者偏差对反转策略的影响（含交易成本）")
plt.xlabel("换仓期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/survivorship_bias.png", dpi=150)
print("\n[OK] 图表已保存 figures/survivorship_bias.png")
