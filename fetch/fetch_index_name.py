# -*- coding: utf-8 -*-
"""
抓取沪深300指数日线 + 全市场股票名称（用于可读的选股清单 + 大盘对比）
- 沪深300：sh.000300 日线收盘
- 名称：query_stock_basic 一次拿全市场 code -> 名称，再对齐到 618 只
输出 index_name_cache.pkl = {"index": Series(date), "name": Series(code->名称)}
"""
import pickle
import pandas as pd
import baostock as bs

CACHE = "index_name_cache.pkl"
START, END = "2015-01-01", "2026-08-14"

lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

# 1) 沪深300 指数
rs = bs.query_history_k_data_plus(
    "sh.000300", "date,close", start_date=START, end_date=END, frequency="d")
rows = []
while rs.next():
    rows.append(rs.get_row_data())
idx = pd.DataFrame(rows, columns=rs.fields)
idx['date'] = pd.to_datetime(idx['date'])
idx = idx.set_index('date')
index_close = pd.to_numeric(idx['close'], errors='coerce').rename('hs300')

# 2) 全市场股票名称
rs2 = bs.query_stock_basic()
fields = rs2.fields
name_map = {}
while rs2.next():
    r = rs2.get_row_data()
    d = dict(zip(fields, r))
    name_map[d.get('code')] = d.get('code_name', d.get('name', ''))
name_all = pd.Series(name_map)

# 3) 对齐到 618 只
with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
codes = list(listed['close'].columns)
name_618 = name_all.reindex(codes)

with open(CACHE, 'wb') as f:
    pickle.dump({"index": index_close, "name": name_618}, f)

print(f"沪深300: {index_close.index[0].date()} ~ {index_close.index[-1].date()} "
      f"共 {len(index_close)} 天，末值 {index_close.iloc[-1]:.0f}")
print(f"名称覆盖: {name_618.notna().sum()}/{len(codes)} 只")
if name_618.notna().sum() < len(codes):
    print(f"  缺失名称样例: {list(name_618[name_618.isna()].index[:5])}")
bs.logout()
