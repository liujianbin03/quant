# -*- coding: utf-8 -*-
"""
龙头股对比分析：拉近3个月日线，算多周期涨跌幅 + 归一化对比图
用法：python stock_compare.py
"""
import time
import akshare as ak
import pandas as pd

# 图表后端 + 中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 关注的龙头股：代码 -> 名称
STOCKS = {
    "600487": "亨通光电",
    "600522": "中天科技",
    "300394": "天孚通信",
    "002371": "北方华创",
    "600584": "长电科技",
    "000636": "风华高科",
}


def sina_symbol(code):
    """6开头->sh，0/3开头->sz"""
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_daily(code, start="20260501", end="20260815"):
    """拉个股日线，带重试"""
    sym = sina_symbol(code)
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date=start,
                                     end_date=end, adjust="qfq")
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print(f"  [{code}] 重试{attempt+1}/3 ({type(e).__name__})")
            time.sleep(2)
    return None


def period_return(close, days):
    """区间涨跌幅：取最近 days 个交易日"""
    if len(close) < days:
        return float('nan')
    return (close.iloc[-1] / close.iloc[-days] - 1) * 100


# ============ 拉数据 ============
print("拉取数据中...")
rows = []
series = {}   # code -> 归一化序列，用于画图

for code, name in STOCKS.items():
    df = fetch_daily(code)
    if df is None or len(df) < 10:
        print(f"[跳过] {name}({code}) 数据不足")
        continue
    df = df.sort_values('date')
    close = df['close'].astype(float)

    r5 = period_return(close, 5)     # 近1周
    r22 = period_return(close, 22)   # 近1月
    r66 = period_return(close, 66)   # 近3月

    rows.append({
        "代码": code, "名称": name,
        "现价": round(close.iloc[-1], 2),
        "近1周%": round(r5, 2),
        "近1月%": round(r22, 2),
        "近3月%": round(r66, 2),
    })
    # 归一化到 100（全区间）
    series[code] = (close / close.iloc[0] * 100)
    print(f"[OK] {name}({code}) 近1月 {r22:+.2f}% 近3月 {r66:+.2f}%")

# ============ 汇总表 ============
res = pd.DataFrame(rows)
print("\n" + "=" * 60)
print("多周期涨跌幅对比（近1周 / 近1月 / 近3月）")
print("=" * 60)
print(res.to_string(index=False))

# ============ 画对比图 ============
if series:
    plt.figure(figsize=(14, 8))
    for code, s in series.items():
        name = STOCKS[code]
        # 用近1月表现区分线粗细/颜色深浅：涨得多的更醒目
        r22 = res.loc[res['代码'] == code, '近1月%'].values[0]
        lw = 2.2 if r22 > 0 else 1.3
        plt.plot(s.index, s.values, label=f"{name}({r22:+.1f}%)", linewidth=lw)

    plt.axhline(100, color='gray', linestyle='--', alpha=0.5)
    plt.title("龙头股走势对比（归一化，起点=100）")
    plt.xlabel("日期")
    plt.ylabel("累计净值")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/compare.png', dpi=150)
    print("\n[OK] 对比图已保存 figures/compare.png")
