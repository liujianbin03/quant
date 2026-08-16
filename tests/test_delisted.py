# -*- coding: utf-8 -*-
"""测试 baostock 退市股数据可得性"""
import baostock as bs

lg = bs.login()
print("login:", lg.error_code, lg.error_msg)

rs = bs.query_stock_basic()
all_rows = []
while rs.next():
    all_rows.append(rs.get_row_data())

# type=1 股票
stocks = [r for r in all_rows if r[4] == "1"]
listed = [r for r in stocks if r[5] == "1"]
delisted = [r for r in stocks if r[5] == "0"]

print(f"股票总数(type=1): {len(stocks)}")
print(f"上市中(status=1): {len(listed)}")
print(f"退市(status=0): {len(delisted)}")

print("\n退市股示例（前10只）:")
for r in delisted[:10]:
    print(f"  {r[0]}  {r[1]}  上市{r[2]}  退市{r[3]}")

# 测试拉取一只退市股的日线
if delisted:
    code = delisted[0][0]
    name = delisted[0][1]
    out_date = delisted[0][3]
    print(f"\n测试拉取退市股: {code} {name} (退市日 {out_date})")
    rs2 = bs.query_history_k_data_plus(
        code, "date,close,turn",
        start_date="2015-01-01", end_date="2026-08-14",
        frequency="d", adjustflag="2")
    rows = []
    while rs2.next():
        rows.append(rs2.get_row_data())
    print(f"  拉到 {len(rows)} 条日线")
    if rows:
        print(f"  首条: {rows[0]}")
        print(f"  末条: {rows[-1]}")

bs.logout()
