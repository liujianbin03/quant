# -*- coding: utf-8 -*-
"""
动量因子回测（Jegadeesh-Titman 风格）
- 宇宙：约50只流动性好的大盘股
- 信号：过去6个月(126交易日)涨幅 = 动量
- 分组：按动量分5档，买最高档(赢家) vs 最低档(输家)
- 持有：1个月(21交易日)后换仓
用法：python momentum_factor.py
"""
import time
import numpy as np
import pandas as pd
import akshare as ak

# 图表后端 + 中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 宇宙：约50只大盘股（覆盖金融/消费/医药/新能源/半导体/有色/科技/地产/能源/汽车）
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
LOOKBACK = 126   # 动量回溯期（约6个月）
HOLD = 21        # 持有期（约1个月）


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


def max_drawdown(cum):
    return ((cum - cum.cummax()) / cum.cummax()).min() * 100


def sharpe(period_returns):
    r = pd.Series(period_returns)
    if r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(12)  # 月度收益年化


# ============ 1. 拉数据 ============
print(f"拉取 {len(UNIVERSE)} 只股票数据（新浪源，需约1分钟）...")
prices = {}
ok = 0
for code, name in UNIVERSE.items():
    s = fetch_daily(code)
    if s is not None:
        prices[code] = s
        ok += 1
    else:
        print(f"  [跳过] {name}({code}) 数据不足")
    time.sleep(0.2)

print(f"\n成功获取 {ok}/{len(UNIVERSE)} 只\n")

price_df = pd.DataFrame(prices).sort_index()
dates = price_df.index

# ============ 2. 动量分组回测 ============
results = {"赢家(高动量)": [], "输家(低动量)": [], "等权基准": []}

i = LOOKBACK
while i + HOLD < len(dates):
    t0 = dates[i - LOOKBACK]
    t1 = dates[i]          # 换仓日
    t2 = dates[i + HOLD]   # 持有期末

    # 动量 = 过去6个月涨幅
    mom = price_df.loc[t1] / price_df.loc[t0] - 1
    mom = mom.dropna()
    if len(mom) < 10:
        i += HOLD
        continue

    # 按动量分5档
    q = pd.qcut(mom.rank(method='first'), 5, labels=False)
    winners = mom[q == 4].index   # 最高档
    losers = mom[q == 0].index    # 最低档

    # 未来1个月收益
    fwd = price_df.loc[t2] / price_df.loc[t1] - 1
    results["赢家(高动量)"].append(fwd[winners].mean())
    results["输家(低动量)"].append(fwd[losers].mean())
    results["等权基准"].append(fwd[mom.index].mean())

    i += HOLD

# ============ 3. 汇总 ============
print("=" * 70)
print(f"动量因子回测结果  (回溯{LOOKBACK}交易日 / 持有{HOLD}交易日)")
print(f"换仓期数：{len(results['赢家(高动量)'])} 期")
print("=" * 70)

summary = {}
cum_curves = {}
for name, rets in results.items():
    s = pd.Series(rets)
    cum = (1 + s).cumprod()
    cum_curves[name] = cum
    total = (cum.iloc[-1] - 1) * 100
    annual = ((cum.iloc[-1]) ** (12 / len(s)) - 1) * 100
    dd = max_drawdown(cum)
    sh = sharpe(rets)
    summary[name] = [total, annual, dd, sh]
    print(f"{name:　<12} 累计收益 {total:+7.2f}%  年化 {annual:+7.2f}%  "
          f"最大回撤 {dd:6.2f}%  夏普 {sh:+.2f}")

# 多空组合（赢家 - 输家）
w = pd.Series(results["赢家(高动量)"])
l = pd.Series(results["输家(低动量)"])
wml = (w - l)
wml_cum = (1 + wml).cumprod()
print(f"\n【多空组合 WML】买赢家卖输家：")
print(f"  累计收益 {(wml_cum.iloc[-1]-1)*100:+7.2f}%  年化 "
      f"{((wml_cum.iloc[-1])**(12/len(wml))-1)*100:+7.2f}%  "
      f"最大回撤 {max_drawdown(wml_cum):6.2f}%  夏普 {sharpe(wml):+.2f}")

# ============ 4. 画图 ============
plt.figure(figsize=(14, 8))
colors = {"赢家(高动量)": "red", "输家(低动量)": "green", "等权基准": "gray"}
for name, cum in cum_curves.items():
    plt.plot(cum.index, cum.values, label=name, color=colors[name], linewidth=2)
plt.axhline(1.0, color='black', linestyle='--', alpha=0.4)
plt.title(f"动量因子回测：买过去6个月涨得好的 vs 涨得差的（{len(UNIVERSE)}只大盘股）")
plt.xlabel("换仓期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/momentum_result.png", dpi=150)
print("\n[OK] 图表已保存 figures/momentum_result.png")
