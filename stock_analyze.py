# -*- coding: utf-8 -*-
"""
多维度个股分析：动量 + 量价关系 + 波动率 + 趋势强度
用法：python stock_analyze.py
"""
import time
import numpy as np
import akshare as ak
import pandas as pd

STOCKS = {
    "600487": "亨通光电",
    "600522": "中天科技",
    "300394": "天孚通信",
    "002371": "北方华创",
    "600584": "长电科技",
    "000636": "风华高科",
}


def sina_symbol(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_daily(code, start="20260401", end="20260815"):
    sym = sina_symbol(code)
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date=start,
                                     end_date=end, adjust="qfq")
            if df is not None and len(df) > 0:
                return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            print(f"  [{code}] 重试{attempt+1}/3 ({type(e).__name__})")
            time.sleep(2)
    return None


def period_return(close, days):
    if len(close) < days:
        return float('nan')
    return (close.iloc[-1] / close.iloc[-days] - 1) * 100


def analyze(code, name):
    df = fetch_daily(code)
    if df is None or len(df) < 30:
        return None

    close = df['close'].astype(float)
    vol = df['volume'].astype(float)

    # ---- 1. 动量 ----
    r5 = period_return(close, 5)
    r22 = period_return(close, 22)
    r66 = period_return(close, 66) if len(close) >= 66 else float('nan')

    # ---- 2. 量价关系 ----
    vol5 = vol.iloc[-5:].mean()
    vol20 = vol.iloc[-20:].mean()
    vol_ratio = vol5 / vol20                     # >1 放量，<1 缩量

    # 量价配合：近20日「上涨日平均量 / 下跌日平均量」
    ret = close.pct_change()
    up_mask = ret > 0
    down_mask = ret < 0
    up_vol = vol[up_mask].iloc[-20:].mean() if up_mask.sum() > 0 else np.nan
    down_vol = vol[down_mask].iloc[-20:].mean() if down_mask.sum() > 0 else np.nan
    vol_price_ratio = up_vol / down_vol if (down_vol and down_vol > 0) else np.nan

    # ---- 3. 波动率（风险） ----
    daily_ret = close.pct_change().dropna()
    vol_annual = daily_ret.iloc[-20:].std() * np.sqrt(252) * 100   # 年化波动率%

    # ---- 4. 趋势强度 ----
    ma20 = close.rolling(20).mean().iloc[-1]
    price_above_ma20 = (close.iloc[-1] / ma20 - 1) * 100           # 现价偏离20日均线%
    ma20_slope = (ma20 / close.rolling(20).mean().iloc[-6] - 1) * 100 if len(close) >= 26 else np.nan  # 20日线5日斜率

    # ---- 5. 区间最大回撤 ----
    cummax = close.cummax()
    max_dd = ((close - cummax) / cummax).min() * 100

    return {
        "代码": code, "名称": name, "现价": round(close.iloc[-1], 2),
        "近1周%": round(r5, 1), "近1月%": round(r22, 1), "近3月%": round(r66, 1),
        "量能比": round(vol_ratio, 2),
        "量价配合": round(vol_price_ratio, 2),
        "年化波动%": round(vol_annual, 1),
        "偏离MA20%": round(price_above_ma20, 1),
        "MA20斜率%": round(ma20_slope, 2),
        "区间回撤%": round(max_dd, 1),
    }


print("拉取数据并计算多维指标...\n")
results = []
for code, name in STOCKS.items():
    r = analyze(code, name)
    if r:
        results.append(r)
        # 量价判断
        vpr = r["量价配合"]
        vj = "涨放量跌缩量(健康)" if vpr > 1.1 else ("涨缩量跌放量(背离)" if vpr < 0.9 else "量价中性")
        print(f"[OK] {name}: 量能比{r['量能比']} | 量价配合{r['量价配合']} -> {vj}")

res = pd.DataFrame(results)
print("\n" + "=" * 100)
print("多维度分析汇总")
print("=" * 100)
print(res.to_string(index=False))

# 保存 CSV 方便查看
res.to_csv("analysis.csv", index=False, encoding="utf-8-sig")
print("\n[OK] 结果已保存 analysis.csv")
