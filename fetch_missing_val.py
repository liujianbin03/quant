# -*- coding: utf-8 -*-
"""补拉缺失的 98 只创业板估值数据，合并进 val_cache.pkl"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import baostock as bs

START, END = "2015-01-01", "2026-08-14"
VAL_CACHE = "val_cache.pkl"
FIX_CACHE = "val_fix.pkl"

with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)

all_codes = list(listed['close'].columns)
have_codes = list(val['pe'].columns)
missing = [c for c in all_codes if c not in have_codes]
print(f"缺失 {len(missing)} 只", flush=True)

new_pe, new_pb, new_ps, new_pcf, done = {}, {}, {}, {}, []
if os.path.exists(FIX_CACHE):
    with open(FIX_CACHE, 'rb') as f:
        cp = pickle.load(f)
    new_pe, new_pb, new_ps, new_pcf, done = cp['pe'], cp['pb'], cp['ps'], cp['pcf'], cp['done']
    print(f"[续跑] 恢复 {len(done)} 只", flush=True)

todo = [c for c in missing if c not in done]
fail = 0
t0 = time.time()
lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit
for i, code in enumerate(todo):
    try:
        rs = bs.query_history_k_data_plus(
            code, "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=START, end_date=END, frequency="d", adjustflag="2")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if len(rows) >= 20:
            df = pd.DataFrame(rows, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            new_pe[code] = pd.to_numeric(df['peTTM'], errors='coerce')
            new_pb[code] = pd.to_numeric(df['pbMRQ'], errors='coerce')
            new_ps[code] = pd.to_numeric(df['psTTM'], errors='coerce')
            new_pcf[code] = pd.to_numeric(df['pcfNcfTTM'], errors='coerce')
        else:
            fail += 1
    except Exception:
        fail += 1
    done.append(code)
    n = len(done)
    if n % 50 == 0:
        el = time.time() - t0
        print(f"[进度] {n}/{len(missing)}  失败{fail}  已用{el:.0f}秒", flush=True)
        with open(FIX_CACHE, 'wb') as f:
            pickle.dump({"pe": new_pe, "pb": new_pb, "ps": new_ps,
                         "pcf": new_pcf, "done": done}, f)

print(f"[完成] 新补 {len(new_pe)} 只，失败 {fail} 只", flush=True)

# 合并
pe = pd.concat([val['pe'], pd.DataFrame(new_pe)], axis=1).sort_index()
pb = pd.concat([val['pb'], pd.DataFrame(new_pb)], axis=1).sort_index()
ps = pd.concat([val['ps'], pd.DataFrame(new_ps)], axis=1).sort_index()
pcf = pd.concat([val['pcf'], pd.DataFrame(new_pcf)], axis=1).sort_index()

with open(VAL_CACHE, 'wb') as f:
    pickle.dump({"pe": pe, "pb": pb, "ps": ps, "pcf": pcf}, f)
print(f"合并后 val_cache: {pe.shape[0]}交易日 × {pe.shape[1]}只（应 618）", flush=True)
if os.path.exists(FIX_CACHE):
    os.remove(FIX_CACHE)
bs.logout()
