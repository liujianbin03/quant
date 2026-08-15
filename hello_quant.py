# -*- coding: utf-8 -*-
"""
你的第一个量化工程 —— 双均线策略回测
流程：akshare 拉A股数据 -> backtrader 回测 -> 出结果 + 保存图表 PNG

用法：
    python hello_quant.py

说明：
    - 先跑通，理解「数据 -> 策略 -> 回测 -> 看结果」这条链路
    - 代码里能改的地方：股票代码 CODE、回测区间、均线参数、初始资金
"""
import akshare as ak
import pandas as pd
import backtrader as bt

# 图表用 Agg 后端（不弹窗、直接存文件），必须在 import pyplot 之前设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体配置（Windows 用微软雅黑），否则中文会变方框
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


# ============================================================
# 1. 拉数据（akshare，A股日线，前复权）
# ============================================================
CODE = "600519"      # 股票代码，可换
NAME = "贵州茅台"     # 股票名称


def fetch_data():
    """拉取日线数据，新浪数据源，失败自动重试"""
    import time
    symbol = f"sh{CODE}" if CODE.startswith("6") else f"sz{CODE}"
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date="20200101", end_date="20241231", adjust="qfq")
            if df is not None and len(df) > 0:
                print(f"[OK] 数据拉取成功，共 {len(df)} 条")
                return df
        except Exception as e:
            print(f"[重试] 拉取失败({type(e).__name__})，第{attempt+1}/3次")
            time.sleep(2)
    raise RuntimeError("数据拉取失败，请检查网络后重试")


df = fetch_data()

# 统一列名 + 日期索引（backtrader 需要）
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"已拉取 {NAME}({CODE}) 数据，区间 {df.index[0].date()} ~ {df.index[-1].date()}")


# ============================================================
# 2. 定义策略：双均线金叉买、死叉卖
# ============================================================
class GoldenCross(bt.Strategy):
    params = (('fast', 5), ('slow', 20),)   # 快线5日、慢线20日

    def __init__(self):
        self.fast = bt.ind.SMA(self.data.close, period=self.p.fast)
        self.slow = bt.ind.SMA(self.data.close, period=self.p.slow)
        self.cross = bt.ind.CrossOver(self.fast, self.slow)

        # 记录买卖点，用于后面画图
        self.buy_dates = []    # 买入日期
        self.buy_prices = []   # 买入价格
        self.sell_dates = []   # 卖出日期
        self.sell_prices = []  # 卖出价格

    def next(self):
        # 金叉且空仓 -> 买入
        if not self.position and self.cross > 0:
            self.buy()
            self.buy_dates.append(self.data.datetime.date(0))
            self.buy_prices.append(self.data.close[0])
        # 死叉且有仓位 -> 卖出
        elif self.position and self.cross < 0:
            self.close()
            self.sell_dates.append(self.data.datetime.date(0))
            self.sell_prices.append(self.data.close[0])


# ============================================================
# 3. 组装回测引擎
# ============================================================
cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.addstrategy(GoldenCross)

# 统计器（回撤、夏普比率）
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

# 初始资金 + 佣金
INIT_CASH = 100000.0
cerebro.broker.setcash(INIT_CASH)
cerebro.broker.setcommission(commission=0.0003)   # 万3佣金
# 仓位管理器：每次买入投入 95% 可用资金（默认只买1股，必须手动设置）
cerebro.addsizer(bt.sizers.PercentSizer, percents=95)


# ============================================================
# 4. 运行回测
# ============================================================
print(f"初始资金: {INIT_CASH:,.0f}")
results = cerebro.run()
strategy = results[0]
final = cerebro.broker.getvalue()

print(f"期末资金: {final:,.2f}")
print(f"总收益:   {final - INIT_CASH:+,.2f}  ({((final/INIT_CASH)-1)*100:.2f}%)")

dd = strategy.analyzers.dd.get_analysis()
sharpe = strategy.analyzers.sharpe.get_analysis()
print(f"最大回撤: {dd.max.drawdown:.2f}%")
print(f"夏普比率: {sharpe.get('sharperatio', 0):.2f}")
print(f"交易次数: 买入 {len(strategy.buy_dates)} 次 / 卖出 {len(strategy.sell_dates)} 次")


# ============================================================
# 5. 画图并保存 PNG（价格 + 两条均线 + 买卖点）
# ============================================================
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='收盘价', color='#333', linewidth=1)

# 计算均线画上去
sma5 = df['close'].rolling(5).mean()
sma20 = df['close'].rolling(20).mean()
plt.plot(df.index, sma5, label='MA5', color='#e67e22', linewidth=1)
plt.plot(df.index, sma20, label='MA20', color='#2980b9', linewidth=1)

# 买卖点
if strategy.buy_dates:
    plt.scatter(strategy.buy_dates, strategy.buy_prices,
                marker='^', color='red', s=80, label='买入', zorder=5)
if strategy.sell_dates:
    plt.scatter(strategy.sell_dates, strategy.sell_prices,
                marker='v', color='green', s=80, label='卖出', zorder=5)

plt.title(f"{NAME}({CODE}) 双均线策略回测  收益 {((final/INIT_CASH)-1)*100:.2f}%")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('result.png', dpi=150)
print("[OK] 图表已保存到 result.png")
