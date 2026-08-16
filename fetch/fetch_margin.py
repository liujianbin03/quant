# -*- coding: utf-8 -*-
"""
抓取融资融券余额历史快照（散户杠杆情绪），用于资金流正交性检验
数据源：akshare stock_margin_detail_sse/szse（上交所/深交所官方源）
口径：按调仓日抓取全市场融资余额快照 → margin_cache.pkl = DataFrame(date × baostock_code)
注：沪市代码列=标的证券代码，深市=证券代码；融资融券标的随时间变化，早年仅大票。
用法（项目根目录）：python fetch/fetch_margin.py
"""
import pickle
import time
import numpy as np
import pandas as pd
import akshare as ak

CACHE = "margin_cache.pkl"
START_YEAR = 2019

with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
close = listed['close']
dates = close.index
# 调仓日（每21交易日），2019年起
HOLD = 21
rebal = [i for i in range(0, len(dates), HOLD) if i + HOLD < len(dates) and dates[i].year >= START_YEAR]


def to_bs(code):
    if code.startswith(('6', '9', '68')):
        return 'sh.' + code
    return 'sz.' + code


def fetch_date(date_str):
    out = {}
    for ex, codecol in [('sse', '标的证券代码'), ('szse', '证券代码')]:
        try:
            df = getattr(ak, f'stock_margin_detail_{ex}')(date=date_str)
            for _, r in df.iterrows():
                try:
                    bal = float(r['融资余额'])
                    if bal > 0:
                        out[to_bs(str(r[codecol]))] = bal
                except Exception:
                    pass
        except Exception as e:
            print(f"  [{ex} {date_str}] 失败 {type(e).__name__}", flush=True)
        time.sleep(0.3)
    return out


print(f"抓取融资余额快照：{len(rebal)} 个调仓日（{START_YEAR}年起）", flush=True)
data = {}
t0 = time.time()
for k, i in enumerate(rebal):
    d = dates[i].strftime('%Y%m%d')
    data[d] = fetch_date(d)
    if (k + 1) % 20 == 0:
        print(f"  [{k+1}/{len(rebal)}] {d} 累计{(time.time()-t0):.0f}秒", flush=True)

mdf = pd.DataFrame(data).T          # index=date, columns=code
mdf.index = pd.to_datetime(mdf.index, format='%Y%m%d')
with open(CACHE, 'wb') as f:
    pickle.dump(mdf, f)
print(f"缓存已保存 {CACHE}: {mdf.shape[0]}日 × {mdf.shape[1]}只", flush=True)
