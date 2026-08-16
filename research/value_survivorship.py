# -*- coding: utf-8 -*-
"""
价值策略幸存者偏差检验：加入退市股（真实占比 ~5%），重估价值 alpha
- 假设：低 PE/PB 是成熟盈利公司，退市风险小 → 幸存者偏差应远小于反转
- 处理：退市股持有期退市 fwd=-90%；按不同占比抽样（全部254/每4/每8）
"""
import pickle
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
DELISTED_CACHE = "delisted_cache.pkl"
VAL_CACHE = "val_cache.pkl"
DELISTED_VAL_CACHE = "delisted_val_cache.pkl"

HOLD, N_HOLD = 21, 50
ROUND_TRIP = 0.00025 + 0.001 + 0.00025 + 0.0005 + 0.001  # 0.30%


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def clean(f):
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


def run(close_df, pe_df, pb_df, ps_df, pcf_df, delist_loss):
    dates = close_df.index
    rets, picks = [], 0
    prev_held = set()
    i = 0
    while i + HOLD < len(dates):
        t1, t2 = dates[i], dates[i + HOLD]
        fwd = close_df.loc[t2] / close_df.loc[t1] - 1
        dmask = close_df.loc[t1].notna() & close_df.loc[t2].isna()
        fwd = fwd.copy()
        fwd[dmask] = delist_loss

        score = -(zscore(clean(pe_df.loc[t1])) + zscore(clean(pb_df.loc[t1])) +
                  zscore(clean(ps_df.loc[t1])) + zscore(clean(pcf_df.loc[t1])))
        valid = score.dropna().index
        if len(valid) >= N_HOLD:
            top = score[valid].nlargest(N_HOLD).index
            gross = fwd[top].mean()
            picks += int(dmask[top].sum())
            turnover = 1 - len(set(top) & prev_held) / N_HOLD if prev_held else 1.0
            rets.append(gross - turnover * ROUND_TRIP)
            prev_held = set(top)
        i += HOLD
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    return (cum.iloc[-1] - 1) * 100, (cum.iloc[-1] ** (12 / len(s)) - 1) * 100, picks


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(DELISTED_CACHE, 'rb') as f:
    delisted = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(DELISTED_VAL_CACHE, 'rb') as f:
    dval = pickle.load(f)

close_l = listed['close']
pe_l = val['pe'].reindex(close_l.index)
pb_l = val['pb'].reindex(close_l.index)
ps_l = val['ps'].reindex(close_l.index)
pcf_l = val['pcf'].reindex(close_l.index)

del_codes = list(delisted['close'].columns)
cd_all = delisted['close']
pe_d = dval['pe'].reindex(cd_all.index)
pb_d = dval['pb'].reindex(cd_all.index)
ps_d = dval['ps'].reindex(cd_all.index)
pcf_d = dval['pcf'].reindex(cd_all.index)

print("=" * 72)
print("价值策略幸存者偏差检验（退市损失 -90%）")
print("=" * 72)
print(f"{'退市股抽样':<16}{'退市占比':>10}{'累计收益':>10}{'年化收益':>10}{'踩雷':>6}")
print("-" * 72)

for step, label in [(1, "全部254只"), (4, "每4取1(~64)"), (8, "每8取1(~32)")]:
    sample = del_codes[::step]
    cd = cd_all[sample]
    pe_dd = pe_d[sample].reindex(close_l.index)
    pb_dd = pb_d[sample].reindex(close_l.index)
    ps_dd = ps_d[sample].reindex(close_l.index)
    pcf_dd = pcf_d[sample].reindex(close_l.index)

    close_all = pd.concat([close_l, cd], axis=1).reindex(close_l.index)
    pe_all = pd.concat([pe_l, pe_dd], axis=1)
    pb_all = pd.concat([pb_l, pb_dd], axis=1)
    ps_all = pd.concat([ps_l, ps_dd], axis=1)
    pcf_all = pd.concat([pcf_l, pcf_dd], axis=1)

    ratio = len(sample) / (close_l.shape[1] + len(sample)) * 100
    total, annual, picks = run(close_all, pe_all, pb_all, ps_all, pcf_all, -0.9)
    print(f"{label:<16}{ratio:>9.1f}%{total:>+9.2f}%{annual:>+9.2f}%{picks:>6}")

print("-" * 72)
print("参考：无退市股 累计 +313.74%，年化 +13.56%（价值策略基线）")
print("反转策略对比：无退市 +119.20% → 含退市(5%) +34.97%，年化 +7.64%→+2.85%")
