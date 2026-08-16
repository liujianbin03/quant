# -*- coding: utf-8 -*-
"""
因子有效性分析：IC分析 + 相关性矩阵
- IC(信息系数) = 因子值 与 未来收益 的秩相关(Spearman)
- ICIR = IC均值/IC标准差（稳定性）
- 胜率 = IC方向正确的期数占比
用法：python factor_ic.py
"""
import os
import time
import numpy as np
import pandas as pd
import akshare as ak

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

UNIVERSE = {
    "600036": "招商银行", "000001": "平安银行", "601318": "中国平安",
    "300059": "东方财富", "600030": "中信证券", "601398": "工商银行",
    "600519": "贵州茅台", "000858": "五粮液", "000568": "泸州老窖",
    "600887": "伊利股份", "603288": "海天味业", "002714": "牧原股份",
    "600276": "恒瑞医药", "300760": "迈瑞医疗", "603259": "药明康德",
    "600436": "片仔癀",
    "300750": "宁德时代", "002594": "比亚迪", "601012": "隆基绿能",
    "300274": "阳光电源",
    "002371": "北方华创", "002475": "立讯精密", "002415": "海康威视",
    "000725": "京东方A", "603501": "韦尔股份", "688981": "中芯国际",
    "601899": "紫金矿业", "600111": "北方稀土", "603799": "华友钴业",
    "603993": "洛阳钼业",
    "000063": "中兴通讯", "002230": "科大讯飞", "000977": "浪潮信息",
    "600487": "亨通光电", "600522": "中天科技", "300394": "天孚通信",
    "000002": "万科A", "600048": "保利发展", "600585": "海螺水泥",
    "600031": "三一重工", "000333": "美的集团", "000651": "格力电器",
    "601888": "中国中免",
    "601088": "中国神华", "600900": "长江电力", "601857": "中国石油",
    "600104": "上汽集团", "601633": "长城汽车", "002352": "顺丰控股",
}

START, END = "20230101", "20260815"
LOOKBACK = 126
HOLD = 21
PRICE_CACHE = "price_cache.csv"
FIN_CACHE = "fin_cache.csv"

# 因子：(名称, 期望方向)  +1=越高越好  -1=越低越好
FACTORS = [
    ("动量", +1),
    ("波动率", -1),
    ("ROE", +1),
    ("资产负债率", -1),
    ("净利润增长率", +1),
    ("质量(合成)", +1),
]


def sina_symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_daily(code):
    sym = sina_symbol(code)
    for _ in range(2):
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date=START,
                                     end_date=END, adjust="qfq")
            if df is not None and len(df) > LOOKBACK + HOLD:
                df = df[['date', 'close']].copy()
                df['date'] = pd.to_datetime(df['date'])
                return df.set_index('date')['close']
        except Exception:
            time.sleep(1)
    return None


def fetch_financial(code):
    for _ in range(2):
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2022")
            if df is not None and len(df) > 0:
                cols = ['日期', '净资产收益率(%)', '资产负债率(%)', '净利润增长率(%)']
                df = df[cols].copy()
                df['日期'] = pd.to_datetime(df['日期'])
                return df.set_index('日期').astype(float)
        except Exception:
            time.sleep(1)
    return None


def load_prices():
    if os.path.exists(PRICE_CACHE):
        return pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    prices = {}
    for code in UNIVERSE:
        s = fetch_daily(code)
        if s is not None:
            prices[code] = s
        time.sleep(0.2)
    df = pd.DataFrame(prices).sort_index()
    df.to_csv(PRICE_CACHE)
    return df


def load_financial(price_index):
    if os.path.exists(FIN_CACHE):
        raw = pd.read_csv(FIN_CACHE)
    else:
        rows = []
        for code in UNIVERSE:
            df = fetch_financial(code)
            if df is not None:
                for dt, r in df.iterrows():
                    rows.append({"code": code, "date": dt,
                                 "roe": r["净资产收益率(%)"],
                                 "debt": r["资产负债率(%)"],
                                 "growth": r["净利润增长率(%)"]})
            time.sleep(0.15)
        raw = pd.DataFrame(rows)
        raw.to_csv(FIN_CACHE, index=False)

    # 关键：code 从CSV读回会丢前导零（000001→1），必须归一化回6位字符串
    raw['code'] = raw['code'].astype(str).str.zfill(6)
    raw['date'] = pd.to_datetime(raw['date'])

    def pivot(col):
        p = raw.pivot_table(index='date', columns='code', values=col)
        p = p.sort_index()
        p = p.reindex(p.index.union(price_index)).sort_index().ffill()
        return p.reindex(price_index)

    return pivot('roe'), pivot('debt'), pivot('growth')


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def spearman(a, b):
    """两个Series的秩相关(Spearman)，自动对齐非NaN，无需scipy"""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return np.nan
    # Spearman = 对排名做 Pearson 相关
    return df['a'].rank().corr(df['b'].rank())


# ============ 数据 ============
print("加载数据...")
price_df = load_prices()
daily_ret = price_df.pct_change(fill_method=None)
dates = price_df.index
roe_ff, debt_ff, growth_ff = load_financial(price_df.index)

# ============ 逐期计算因子值 + 未来收益 ============
ic_collect = {name: [] for name, _ in FACTORS}
pool = {name: [] for name, _ in FACTORS}   # 用于相关性矩阵

i = LOOKBACK
periods = 0
while i + HOLD < len(dates):
    t0 = dates[i - LOOKBACK]
    t1 = dates[i]
    t2 = dates[i + HOLD]

    mom = price_df.loc[t1] / price_df.loc[t0] - 1
    vol = daily_ret.loc[t0:t1].std()
    roe_s = roe_ff.loc[t1]
    debt_s = debt_ff.loc[t1]
    growth_s = growth_ff.loc[t1]
    quality = zscore(roe_s) - zscore(debt_s) + zscore(growth_s)

    fwd = price_df.loc[t2] / price_df.loc[t1] - 1

    fmap = {
        "动量": mom, "波动率": vol, "ROE": roe_s,
        "资产负债率": debt_s, "净利润增长率": growth_s, "质量(合成)": quality,
    }

    for name, _ in FACTORS:
        f = fmap[name]
        ic = spearman(f, fwd)
        if not np.isnan(ic):
            ic_collect[name].append(ic)
        # 池化（z-score后）用于相关性矩阵
        pool[name].append(zscore(f))

    i += HOLD
    periods += 1

print(f"共 {periods} 个换仓期\n")

# ============ IC 汇总 ============
print("=" * 78)
print("因子 IC 分析结果（Spearman 秩相关，越高越有效）")
print("=" * 78)
print(f"{'因子':<12}{'IC均值':>8}{'IC标准差':>9}{'ICIR':>8}{'|t值|':>8}{'胜率':>8}")
print("-" * 78)

summary_rows = []
for name, direction in FACTORS:
    s = pd.Series(ic_collect[name]).dropna()
    if len(s) == 0:
        continue
    ic_mean = s.mean()
    ic_std = s.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    t = abs(icir) * np.sqrt(len(s))
    # 胜率：IC方向与期望方向一致的比例
    win = ((s * direction) > 0).mean() * 100
    summary_rows.append([name, ic_mean, ic_std, icir, t, win, len(s)])
    print(f"{name:<12}{ic_mean:>8.4f}{ic_std:>9.4f}{icir:>8.3f}{t:>8.2f}{win:>7.0f}%")

print("-" * 78)
print("说明：ICIR>0.3 算好因子，>0.5 优秀；|t|>2 表示统计显著")
print("      波动率/资产负债率 期望IC为负（低波/低负债更好）\n")

# ============ 相关性矩阵 ============
names = [n for n, _ in FACTORS]
corr_df = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        ca = pd.concat(pool[a]).reset_index(drop=True)
        cb = pd.concat(pool[b]).reset_index(drop=True)
        corr_df.loc[a, b] = ca.corr(cb)

print("=" * 78)
print("因子相关性矩阵（绝对值越大越冗余）")
print("=" * 78)
print(corr_df.round(3).to_string())

# ============ 相关性热力图 ============
plt.figure(figsize=(9, 8))
mat = corr_df.values.astype(float)
im = plt.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1)
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        plt.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                 fontsize=9, color='black')
plt.colorbar(im, label='相关系数')
plt.title("因子相关性矩阵")
plt.tight_layout()
plt.savefig("figures/factor_corr_matrix.png", dpi=150)
print("\n[OK] 相关性热力图已保存 figures/factor_corr_matrix.png")

# ============ IC 时序图 ============
plt.figure(figsize=(14, 6))
for name, _ in FACTORS:
    if ic_collect[name]:
        plt.plot(ic_collect[name], marker='o', label=name, linewidth=1.5)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title("各因子 IC 时序（>0 表示本期因子有效）")
plt.xlabel("换仓期")
plt.ylabel("IC (秩相关)")
plt.legend(ncol=3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/factor_ic_timeseries.png", dpi=150)
print("[OK] IC时序图已保存 figures/factor_ic_timeseries.png")
