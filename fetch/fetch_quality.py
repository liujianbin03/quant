# -*- coding: utf-8 -*-
"""一次拉取多质量因子（618 只上市股年报），带断点续跑
数据源：新浪 ak.stock_financial_analysis_indicator（一次调用返回全部季度字段）
提取每年 12-31 的年报值，存 quality_cache.pkl = {因子名: DataFrame(year×code)}
因子（方向）：
  毛利率   = 销售毛利率(%)          高好
  净利增速 = 净利润增长率(%)        高好
  营收增速 = 主营业务收入增长率(%)  高好
  负债率   = 资产负债率(%)          低好
  ROE     = 净资产收益率(%)        高好（用于算稳定性）
"""
import os
import pickle
import time
import pandas as pd
import akshare as ak

CACHE = "quality_cache.pkl"
CHECKPOINT = "quality_checkpoint.pkl"

FACTORS = {
    "毛利率": "销售毛利率(%)",
    "净利增速": "净利润增长率(%)",
    "营收增速": "主营业务收入增长率(%)",
    "负债率": "资产负债率(%)",
    "ROE": "净资产收益率(%)",
}


def get_annual(code):
    """返回 {year: {因子名: value}}"""
    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2014")
    if df is None or len(df) == 0:
        return {}
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期")
    out = {}
    for d, row in df.iterrows():
        if d.month == 12 and d.day == 31:
            y = d.year
            out[y] = {}
            for name, col in FACTORS.items():
                try:
                    out[y][name] = float(row[col])
                except Exception:
                    out[y][name] = float('nan')
    return out


with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
CODES = [c.split('.')[1] for c in listed['close'].columns]

data, done = {}, []  # data: {code: {year: {factor: value}}}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, 'rb') as f:
        cp = pickle.load(f)
    data, done = cp['data'], cp['done']
    print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

todo = [c for c in CODES if c not in done]
fail = 0
t0 = time.time()
for i, code in enumerate(todo):
    time.sleep(0.3)  # 限速：避免高频请求被封 IP（新浪源）
    try:
        annual = get_annual(code)
        if annual:
            data[code] = annual
        else:
            fail += 1
    except Exception:
        fail += 1
    done.append(code)
    n = len(done)
    if n % 100 == 0:
        el = time.time() - t0
        speed = (i + 1) / el if el > 0 else 0
        print(f"[进度] {n}/{len(CODES)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump({"data": data, "done": done}, f)

print(f"[完成] 成功 {len(data)} 只，失败 {fail} 只", flush=True)

# 转成 dict of DataFrame：{因子名: DataFrame(index=year, columns=code)}
cache = {}
for name in FACTORS:
    d = {code: {y: v[name] for y, v in years.items()} for code, years in data.items()}
    df = pd.DataFrame.from_dict(d, orient='index').T
    cache[name] = df.sort_index()

with open(CACHE, 'wb') as f:
    pickle.dump(cache, f)
print(f"质量缓存已保存 {CACHE}: 每个因子 {cache['ROE'].shape[0]}年 × {cache['ROE'].shape[1]}只", flush=True)
print(f"年份范围: {cache['ROE'].index.min()} - {cache['ROE'].index.max()}", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
