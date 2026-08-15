# -*- coding: utf-8 -*-
"""退市股数据可得性 v2：只看 2015 之后退市的（才是幸存者偏差来源）"""
import baostock as bs

lg = bs.login()
rs = bs.query_stock_basic()
rows = []
while rs.next():
    rows.append(rs.get_row_data())

stocks = [r for r in rows if r[4] == "1"]
delisted = [r for r in stocks if r[5] == "0"]

# 只看 2015 之后退市的（2015 前的本来就不在我们的样本期里）
recent = [r for r in delisted if r[3] >= "2015-01-01"]
print(f"退市股总数: {len(delisted)}")
print(f"2015 后退市（影响我们样本期的）: {len(recent)}")
print("\n2015 后退市股示例:")
for r in recent[:15]:
    print(f"  {r[0]}  {r[1]}  退市{r[3]}")

# 测试拉取 2015 后退市的股票（葛洲坝 sh.600068，退市 2021-09-13）
print("\n=== 测试拉取葛洲坝 sh.600068 (退市2021) ===")
rs2 = bs.query_history_k_data_plus(
    "sh.600068", "date,close,turn",
    start_date="2015-01-01", end_date="2026-08-14",
    frequency="d", adjustflag="2")
r2 = []
while rs2.next():
    r2.append(rs2.get_row_data())
print(f"拉到 {len(r2)} 条")
if r2:
    print(f"首条: {r2[0]}")
    print(f"末条: {r2[-1]}")

# 测试一只 2025 年才退市的（*ST富润 sh.600070）
print("\n=== 测试拉取 *ST富润 sh.600070 (退市2025-04) ===")
rs3 = bs.query_history_k_data_plus(
    "sh.600070", "date,close,turn",
    start_date="2015-01-01", end_date="2026-08-14",
    frequency="d", adjustflag="2")
r3 = []
while rs3.next():
    r3.append(rs3.get_row_data())
print(f"拉到 {len(r3)} 条")
if r3:
    print(f"首条: {r3[0]}")
    print(f"末条: {r3[-1]}")

bs.logout()
