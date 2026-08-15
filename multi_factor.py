# -*- coding: utf-8 -*-
"""
多因子打分回测：动量 + 低波动
- 动量因子：过去6个月涨幅（越高越好）
- 低波动因子：过去6个月日收益波动率（越低越好）
- 合成：两个因子标准化后相加 -> 选"涨得稳"的股票
对比 4 个组合：纯动量 / 纯低波动 / 多因子 / 等权基准
用法：python multi_factor.py
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
LOOKBACK = 126   # 因子回溯期（约6个月）
HOLD = 21        # 持有期（约1个月）
CACHE = "price_cache.csv"


def sina_symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_daily(code):
    sym = sina_symbol(code)
    for attempt in range(2):
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


def load_prices():
    """带缓存：有缓存直接读，否则拉取并保存"""
    if os.path.exists(CACHE):
        print("读取缓存价格数据...")
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        return df
    print(f"拉取 {len(UNIVERSE)} 只股票数据（约1分钟）...")
    prices = {}
    for code in UNIVERSE:
        s = fetch_daily(code)
        if s is not None:
            prices[code] = s
        time.sleep(0.2)
    df = pd.DataFrame(prices).sort_index()
    df.to_csv(CACHE)
    print(f"成功获取 {df.shape[1]} 只，已缓存")
    return df


def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0, index=s.index)


def max_drawdown(cum):
    return ((cum - cum.cummax()) / cum.cummax()).min() * 100


def sharpe(period_returns):
    r = pd.Series(period_returns)
    if r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(12)


# ============ 数据 ============
price_df = load_prices()
daily_ret = price_df.pct_change()
dates = price_df.index

# ============ 回测 ============
PORTFOLIOS = ["纯动量", "纯低波动", "多因子(动量+低波)", "等权基准"]
results = {p: [] for p in PORTFOLIOS}

i = LOOKBACK
while i + HOLD < len(dates):
    t0 = dates[i - LOOKBACK]
    t1 = dates[i]
    t2 = dates[i + HOLD]

    # 动量因子
    mom = price_df.loc[t1] / price_df.loc[t0] - 1
    # 低波动因子（过去6个月日收益标准差，越小越好）
    vol = daily_ret.loc[t0:t1].std()

    valid = mom.dropna().index.intersection(vol.dropna().index)
    if len(valid) < 10:
        i += HOLD
        continue

    # 标准化（横截面 z-score）
    z_mom = zscore(mom[valid])
    z_vol = zscore(vol[valid])
    # 多因子打分：高动量(+z_mom) + 低波动(-z_vol)
    score = z_mom - z_vol

    # 分5档取最优档
    def top(s, ascending):
        q = pd.qcut(s.rank(method='first'), 5, labels=False)
        tgt = 0 if ascending else 4
        return s[q == tgt].index

    mom_win = top(mom[valid], ascending=False)   # 动量最高
    vol_win = top(vol[valid], ascending=True)    # 波动最低
    mf_win = top(score, ascending=False)         # 综合分最高

    fwd = price_df.loc[t2] / price_df.loc[t1] - 1

    results["纯动量"].append(fwd[mom_win].mean())
    results["纯低波动"].append(fwd[vol_win].mean())
    results["多因子(动量+低波)"].append(fwd[mf_win].mean())
    results["等权基准"].append(fwd[valid].mean())

    i += HOLD

# ============ 汇总 ============
print("=" * 72)
print(f"多因子回测结果（动量+低波动） 回溯{LOOKBACK}日/持有{HOLD}日")
print(f"换仓期数：{len(results['纯动量'])} 期")
print("=" * 72)

cum_curves = {}
for name in PORTFOLIOS:
    s = pd.Series(results[name])
    cum = (1 + s).cumprod()
    cum_curves[name] = cum
    total = (cum.iloc[-1] - 1) * 100
    annual = ((cum.iloc[-1]) ** (12 / len(s)) - 1) * 100
    print(f"{name:　<16} 累计 {total:+7.2f}%  年化 {annual:+7.2f}%  "
          f"回撤 {max_drawdown(cum):6.2f}%  夏普 {sharpe(s):+.2f}")

# 因子相关性
valid_all = mom.dropna().index.intersection(vol.dropna().index)
corr = mom[valid_all].corr(vol[valid_all])
print(f"\n动量因子 vs 低波动因子 的相关性：{corr:.3f}  "
      f"（越接近0说明两个因子越互补）")

# ============ 画图 ============
plt.figure(figsize=(14, 8))
colors = {"纯动量": "orange", "纯低波动": "blue",
          "多因子(动量+低波)": "red", "等权基准": "gray"}
for name, cum in cum_curves.items():
    plt.plot(cum.index, cum.values, label=name, color=colors[name], linewidth=2.2)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.4)
plt.title("多因子对比：纯动量 vs 纯低波动 vs 多因子组合 vs 等权基准")
plt.xlabel("换仓期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("multi_factor_result.png", dpi=150)
print("\n[OK] 图表已保存 multi_factor_result.png")
