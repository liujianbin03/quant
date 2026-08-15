# -*- coding: utf-8 -*-
"""拉取 254 只退市股的估值因子（PE/PB/PS/PCF），带断点续跑"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import baostock as bs

START, END = "2015-01-01", "2026-08-14"
CACHE = "delisted_val_cache.pkl"
CHECKPOINT = "delisted_val_checkpoint.pkl"

with open("delisted_cache.pkl", 'rb') as f:
    delisted = pickle.load(f)
CODES = list(delisted['close'].columns)


def fetch(codes):
    pe_dict, pb_dict, ps_dict, pcf_dict, done = {}, {}, {}, {}, []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        pe_dict, pb_dict, ps_dict, pcf_dict, done = cp['pe'], cp['pb'], cp['ps'], cp['pcf'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in codes if c not in done]
    fail = 0
    t0 = time.time()
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
                pe_dict[code] = pd.to_numeric(df['peTTM'], errors='coerce')
                pb_dict[code] = pd.to_numeric(df['pbMRQ'], errors='coerce')
                ps_dict[code] = pd.to_numeric(df['psTTM'], errors='coerce')
                pcf_dict[code] = pd.to_numeric(df['pcfNcfTTM'], errors='coerce')
            else:
                fail += 1
        except Exception:
            fail += 1
        done.append(code)
        n = len(done)
        if n % 50 == 0:
            el = time.time() - t0
            speed = (i + 1) / el if el > 0 else 0
            print(f"[进度] {n}/{len(codes)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"pe": pe_dict, "pb": pb_dict, "ps": ps_dict,
                             "pcf": pcf_dict, "done": done}, f)

    print(f"[完成] 成功 {len(pe_dict)} 只，失败 {fail} 只", flush=True)
    return pe_dict, pb_dict, ps_dict, pcf_dict


lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

print(f"待拉取退市股估值: {len(CODES)} 只", flush=True)
pe, pb, ps, pcf = fetch(CODES)

pe_df = pd.DataFrame(pe).sort_index()
pb_df = pd.DataFrame(pb).sort_index()
ps_df = pd.DataFrame(ps).sort_index()
pcf_df = pd.DataFrame(pcf).sort_index()

with open(CACHE, 'wb') as f:
    pickle.dump({"pe": pe_df, "pb": pb_df, "ps": ps_df, "pcf": pcf_df}, f)
print(f"退市股估值缓存已保存 {CACHE}: {pe_df.shape[0]}交易日 × {pe_df.shape[1]}只", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
bs.logout()
