# -*- coding: utf-8 -*-
"""测试 baostock 连接 + 股票列表 + 日线数据"""
import baostock as bs

lg = bs.login()
print(f"login: code={lg.error_code} msg={lg.error_msg}")

# 1) 全市场股票列表
rs = bs.query_stock_basic()
stocks = []
while rs.next():
    stocks.append(rs.get_row_data())
print(f"stock_basic 总条数: {len(stocks)}")

# 统计类型
from collections import Counter
import pandas as pd
df = pd.DataFrame(stocks, columns=rs.fields)
print("字段:", rs.fields)
print("type分布:", dict(Counter(df['type'])))
print("status分布:", dict(Counter(df['status'])))
print("\n样例:")
print(df.head(5).to_string())

# 2) 单只股票日线（前复权）
rs2 = bs.query_history_k_data_plus(
    "sh.600519", "date,code,close,volume,pctChg,turn",
    start_date="2026-01-01", end_date="2026-08-15",
    frequency="d", adjustflag="2")
rows = []
while rs2.next():
    rows.append(rs2.get_row_data())
kdf = pd.DataFrame(rows, columns=rs2.fields)
print(f"\n茅台日线前复权: {len(kdf)} 行")
print(kdf.tail(3).to_string())

# 3) 财务数据（盈利能力）
rs3 = bs.query_profit_data(code="sh.600519", year=2025, quarter=4)
prows = []
while rs3.next():
    prows.append(rs3.get_row_data())
pdf = pd.DataFrame(prows, columns=rs3.fields)
print(f"\n茅台2025年报盈利指标: {len(pdf)} 行")
if len(pdf):
    print("关键字段:", [c for c in pdf.columns if 'ROE' in c or 'roe' in c or 'NetProfit' in c or 'EPS' in c])

bs.logout()
print("\n[OK] 测试完成")
