# -*- coding: utf-8 -*-
"""
策略擂台：5 个经典开源策略，跑同一只票，对比表现
这些策略都是几十年前公开的经典策略，backtrader 自带指标，直接复用
用法：python strategy_battle.py
"""
import time
import akshare as ak
import backtrader as bt

CODE = "000636"      # 风华高科
NAME = "风华高科"
START, END = "20230101", "20251231"


def sina_symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_daily(code, start, end):
    sym = sina_symbol(code)
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date=start,
                                     end_date=end, adjust="qfq")
            if df is not None and len(df) > 30:
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df['date'] = __import__('pandas').to_datetime(df['date'])
                df = df.set_index('date')
                return df
        except Exception as e:
            print(f"  [{code}] 重试{attempt+1}/3 ({type(e).__name__})")
            time.sleep(2)
    return None


# ============ 5 个经典策略（都是公开的经典策略，指标用 backtrader 自带的） ============

class SmaCross(bt.Strategy):
    """1. 双均线金叉买死叉卖"""
    params = (('fast', 5), ('slow', 20),)
    def __init__(self):
        self.cross = bt.ind.CrossOver(
            bt.ind.SMA(self.data.close, period=self.p.fast),
            bt.ind.SMA(self.data.close, period=self.p.slow))
    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()


class MacdCross(bt.Strategy):
    """2. MACD 金叉买死叉卖"""
    def __init__(self):
        macd = bt.ind.MACD(self.data.close)
        self.cross = bt.ind.CrossOver(macd.macd, macd.signal)
    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()


class RsiReversal(bt.Strategy):
    """3. RSI 超卖买(30)、超买卖(70) 均值回归"""
    def __init__(self):
        self.rsi = bt.ind.RSI(self.data.close, period=14)
    def next(self):
        if not self.position and self.rsi < 30:
            self.buy()
        elif self.position and self.rsi > 70:
            self.close()


class BollReversal(bt.Strategy):
    """4. 布林带：跌破下轨买、突破上轨卖 均值回归"""
    def __init__(self):
        self.bb = bt.ind.BollingerBands(self.data.close, period=20, devfactor=2)
    def next(self):
        if not self.position and self.data.close < self.bb.bot:
            self.buy()
        elif self.position and self.data.close > self.bb.top:
            self.close()


class Donchian(bt.Strategy):
    """5. 海龟/唐奇安通道：突破20日高点买、跌破10日低点卖 趋势跟踪"""
    def __init__(self):
        self.hh = bt.ind.Highest(self.data.high, period=20)
        self.ll = bt.ind.Lowest(self.data.low, period=10)
    def next(self):
        if not self.position and self.data.close > self.hh[-1]:
            self.buy()
        elif self.position and self.data.close < self.ll[-1]:
            self.close()


STRATEGIES = [
    ("双均线交叉", SmaCross),
    ("MACD金叉", MacdCross),
    ("RSI超买超卖", RsiReversal),
    ("布林带回归", BollReversal),
    ("唐奇安通道", Donchian),
]


def run_one(strategy_cls, data):
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_cls)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0003)
    # 关键：设置仓位管理器，每次买入投入 95% 可用资金（默认只买1股）
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    res = cerebro.run()
    s = res[0]
    final = cerebro.broker.getvalue()
    dd = s.analyzers.dd.get_analysis().max.drawdown
    sharpe = s.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    ta = s.analyzers.trade.get_analysis()
    trades = ta.get('total', {}).get('closed', 0)
    return {
        "收益%": round((final/100000 - 1) * 100, 2),
        "最大回撤%": round(dd, 2),
        "夏普": round(sharpe if sharpe else 0, 2),
        "交易次数": trades,
    }


print(f"回测标的：{NAME}({CODE})  {START}~{END}\n")
df = fetch_daily(CODE, START, END)
if df is None:
    print("数据拉取失败")
    exit(1)
print(f"数据 {len(df)} 条\n")

import pandas as pd
rows = []
for name, cls in STRATEGIES:
    # 每个策略用独立的数据源实例（避免状态污染）
    data = bt.feeds.PandasData(dataname=df.copy())
    r = run_one(cls, data)
    r["策略"] = name
    rows.append(r)
    print(f"[OK] {name}: 收益{r['收益%']}% | 回撤{r['最大回撤%']}% | 夏普{r['夏普']} | 交易{r['交易次数']}次")

res = pd.DataFrame(rows)[["策略", "收益%", "最大回撤%", "夏普", "交易次数"]]
print("\n" + "=" * 60)
print("策略表现对比")
print("=" * 60)
print(res.to_string(index=False))
