# -*- coding: utf-8 -*-
"""拉取 618 只上市股的年报 ROE（净资产收益率），带断点续跑
- 数据源：新浪 ak.stock_financial_analysis_indicator（一次调用返回全部季度）
- 提取每年 12-31 的「净资产收益率(%)」= 年度 ROE（YTD 累计值）
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import akshare as ak

CACHE = "roe_cache.pkl"
CHECKPOINT = "roe_checkpoint.pkl"


def get_annual_roe(code):
    """返回 {year: annual_roe} 字典"""
    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2014")
    if df is None or len(df) == 0:
        return {}
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期")
    if "净资产收益率(%)" not in df.columns:
        return {}
    roe = df["净资产收益率(%)"]
    annual = {}
    for d, v in roe.items():
        if d.month == 12 and d.day == 31:
            annual[d.year] = float(v)
    return annual


with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
CODES = [c.split('.')[1] for c in listed['close'].columns]  # sh.600519 -> 600519

roe_dict, done = {}, []
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, 'rb') as f:
        cp = pickle.load(f)
    roe_dict, done = cp['roe'], cp['done']
    print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

todo = [c for c in CODES if c not in done]
fail = 0
t0 = time.time()
for i, code in enumerate(todo):
    try:
        annual = get_annual_roe(code)
        if annual:
            roe_dict[code] = annual
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
            pickle.dump({"roe": roe_dict, "done": done}, f)

print(f"[完成] 成功 {len(roe_dict)} 只，失败 {fail} 只", flush=True)

# 转成 DataFrame：index=year, columns=code
roe_df = pd.DataFrame.from_dict(roe_dict, orient='index').T  # index=year
roe_df = roe_df.sort_index()
with open(CACHE, 'wb') as f:
    pickle.dump(roe_df, f)
print(f"ROE 缓存已保存 {CACHE}: {roe_df.shape[0]}年 × {roe_df.shape[1]}只", flush=True)
print(f"年份范围: {roe_df.index.min()} - {roe_df.index.max()}", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
