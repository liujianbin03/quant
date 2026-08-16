# -*- coding: utf-8 -*-
"""快速验证 baostock 月线 + 估值字段可用"""
import baostock as bs
import pandas as pd

lg = bs.login()
rs = bs.query_history_k_data_plus(
    "sh.600036", "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn",
    start_date="2015-01-01", end_date="2026-08-14",
    frequency="d", adjustflag="2")
rows = []
while rs.next():
    rows.append(rs.get_row_data())
df = pd.DataFrame(rows, columns=rs.fields)
print("日线行数:", len(df))
print("列:", list(df.columns))
print("--- 前3行 ---")
print(df.head(3).to_string())
print("--- 后2行 ---")
print(df.tail(2).to_string())
print("--- 非空统计 ---")
print(df[['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']].count())
bs.logout()
