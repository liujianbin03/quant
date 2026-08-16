# -*- coding: utf-8 -*-
"""
抓取 618 只的年度每股派息(税前 dividCashPsBeforeTax)，用于计算股息率(carry因子)
数据源：baostock query_dividend_data(code, year, yearType="report")
口径：每年可能有多次派息，取该年税前派息之和 → dividend_cache.pkl = DataFrame(year × code)
用法（项目根目录）：python fetch/fetch_dividend.py
"""
import os
import pickle
import time
import pandas as pd
import baostock as bs

CACHE = "dividend_cache.pkl"
CHECKPOINT = "dividend_checkpoint.pkl"
YEARS = list(range(2015, 2026))

with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
CODES = list(listed['close'].columns)


def fetch(codes):
    dps_dict, done = {}, []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        dps_dict, done = cp['dps'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in codes if c not in done]
    t0 = time.time()
    fail = 0
    for i, code in enumerate(todo):
        yearly = {}
        for year in YEARS:
            try:
                rs = bs.query_dividend_data(code=code, year=str(year), yearType="report")
                s = 0.0
                while rs.next():
                    r = rs.get_row_data()
                    d = dict(zip(rs.fields, r))
                    v = d.get('dividCashPsBeforeTax', '')
                    if v not in (None, '', '0', '0.000000'):
                        try:
                            s += float(v)
                        except Exception:
                            pass
                if s > 0:
                    yearly[year] = round(s, 4)
            except Exception:
                fail += 1
            time.sleep(0.25)   # 限速：避免高频请求被封 IP
        dps_dict[code] = yearly
        done.append(code)
        n = len(done)
        if n % 50 == 0:
            el = time.time() - t0
            print(f"[进度] {n}/{len(codes)}  失败{fail}  {n/el:.2f}只/秒  已用{el:.0f}秒", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"dps": dps_dict, "done": done}, f)

    print(f"[完成] 有派息 {sum(1 for v in dps_dict.values() if v)} 只，失败 {fail} 只", flush=True)
    return dps_dict


lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

print(f"待抓取股息: {len(CODES)} 只 × {len(YEARS)} 年", flush=True)
dps = fetch(CODES)

dps_df = pd.DataFrame(dps).sort_index()   # index=year, columns=code
with open(CACHE, 'wb') as f:
    pickle.dump(dps_df, f)
print(f"缓存已保存 {CACHE}: {dps_df.shape[0]}年 × {dps_df.shape[1]}只", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
bs.logout()
