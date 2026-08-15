# -*- coding: utf-8 -*-
"""
三因子打分回测：动量 + 低波动 + 质量
- 动量：过去6个月涨幅（越高越好）
- 低波动：过去6个月日收益波动率（越低越好）
- 质量：ROE(高) + 资产负债率(低) + 净利润增长率(高)
对比 5 个组合：纯动量/纯低波动/纯质量/三因子/等权基准
用法：python multi_factor3.py
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
                df = df.set_index('日期').astype(float)
                return df
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
    """拉取并缓存财务数据，返回 (roe_ff, debt_ff, growth_ff) 已对齐到交易日历"""
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

    raw['date'] = pd.to_datetime(raw['date'])

    def pivot(col):
        p = raw.pivot_table(index='date', columns='code', values=col)
        p = p.sort_index()
        # 对齐到交易日历并前向填充（point-in-time：只用当天及之前已公布的数据）
        p = p.reindex(p.index.union(price_index)).sort_index().ffill()
        return p.reindex(price_index)

    return pivot('roe'), pivot('debt'), pivot('growth')


def zscore_robust(s):
    """稳健z-score：先截尾到5%~95%分位再标准化"""
    s = s.astype(float)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    s = s.clip(lo, hi)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def max_drawdown(cum):
    return ((cum - cum.cummax()) / cum.cummax()).min() * 100


def sharpe(period_returns):
    r = pd.Series(period_returns)
    if r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(12)


# ============ 数据 ============
print("加载价格数据...")
price_df = load_prices()
daily_ret = price_df.pct_change(fill_method=None)
dates = price_df.index

print("加载财务数据（约1-2分钟）...")
roe_ff, debt_ff, growth_ff = load_financial(price_df.index)
print(f"财务数据覆盖 {roe_ff.notna().sum().sum()} 个 (股票×季度) 数据点\n")

# ============ 回测 ============
PORTFOLIOS = ["纯动量", "纯低波动", "纯质量", "三因子(动量+低波+质量)", "等权基准"]
results = {p: [] for p in PORTFOLIOS}

i = LOOKBACK
while i + HOLD < len(dates):
    t0 = dates[i - LOOKBACK]
    t1 = dates[i]
    t2 = dates[i + HOLD]

    mom = price_df.loc[t1] / price_df.loc[t0] - 1
    vol = daily_ret.loc[t0:t1].std()

    roe_s = roe_ff.loc[t1]
    debt_s = debt_ff.loc[t1]
    growth_s = growth_ff.loc[t1]
    quality = zscore_robust(roe_s) - zscore_robust(debt_s) + zscore_robust(growth_s)

    valid = mom.dropna().index.intersection(vol.dropna().index)\
                        .intersection(quality.dropna().index)
    if len(valid) < 10:
        i += HOLD
        continue

    z_mom = zscore_robust(mom[valid])
    z_vol = zscore_robust(vol[valid])
    q = quality[valid]
    score3 = z_mom - z_vol + q   # 三因子合成

    def top(s, ascending):
        qq = pd.qcut(s.rank(method='first'), 5, labels=False)
        tgt = 0 if ascending else 4
        return s[qq == tgt].index

    fwd = price_df.loc[t2] / price_df.loc[t1] - 1

    results["纯动量"].append(fwd[top(mom[valid], False)].mean())
    results["纯低波动"].append(fwd[top(vol[valid], True)].mean())
    results["纯质量"].append(fwd[top(q, False)].mean())
    results["三因子(动量+低波+质量)"].append(fwd[top(score3, False)].mean())
    results["等权基准"].append(fwd[valid].mean())

    i += HOLD

# ============ 汇总 ============
print("=" * 78)
print(f"三因子回测结果（动量+低波动+质量）  回溯{LOOKBACK}日/持有{HOLD}日")
print(f"换仓期数：{len(results['纯动量'])} 期")
print("=" * 78)

cum_curves = {}
for name in PORTFOLIOS:
    s = pd.Series(results[name])
    cum = (1 + s).cumprod()
    cum_curves[name] = cum
    total = (cum.iloc[-1] - 1) * 100
    annual = ((cum.iloc[-1]) ** (12 / len(s)) - 1) * 100
    print(f"{name:　<22} 累计 {total:+7.2f}%  年化 {annual:+7.2f}%  "
          f"回撤 {max_drawdown(cum):6.2f}%  夏普 {sharpe(s):+.2f}")

# ============ 画图 ============
plt.figure(figsize=(14, 8))
colors = {"纯动量": "orange", "纯低波动": "blue", "纯质量": "purple",
          "三因子(动量+低波+质量)": "red", "等权基准": "gray"}
for name, cum in cum_curves.items():
    lw = 2.6 if "三因子" in name else 2.0
    plt.plot(cum.index, cum.values, label=name, color=colors[name], linewidth=lw)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.4)
plt.title("三因子对比：动量 / 低波动 / 质量 / 三因子组合 / 等权基准")
plt.xlabel("换仓期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("multi_factor3_result.png", dpi=150)
print("\n[OK] 图表已保存 multi_factor3_result.png")
