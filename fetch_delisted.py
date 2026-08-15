# -*- coding: utf-8 -*-
"""拉取 2015 后退市的股票日线（带断点续跑），用于幸存者偏差分析"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import baostock as bs

START, END = "2015-01-01", "2026-08-14"
CACHE = "delisted_cache.pkl"
CHECKPOINT = "delisted_checkpoint.pkl"


def get_delisted():
    rs = bs.query_stock_basic()
    out = []
    while rs.next():
        r = rs.get_row_data()
        # type=1 股票, status=0 退市, 2015 之后退市
        if r[4] == "1" and r[5] == "0" and r[3] >= "2015-01-01":
            out.append((r[0], r[3]))
    return out


def fetch(delisted):
    close_dict, turn_dict, out_dict, done = {}, {}, {}, []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        close_dict, turn_dict, out_dict, done = cp['close'], cp['turn'], cp['out'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [x for x in delisted if x[0] not in done]
    fail = 0
    t0 = time.time()
    for i, (code, out_date) in enumerate(todo):
        time.sleep(0.5)  # 限速：避免高频请求被 baostock 封 IP
        try:
            rs = bs.query_history_k_data_plus(
                code, "date,close,turn",
                start_date=START, end_date=END,
                frequency="d", adjustflag="2")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if len(rows) >= 20:
                df = pd.DataFrame(rows, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                close_dict[code] = pd.to_numeric(df['close'], errors='coerce')
                turn_dict[code] = pd.to_numeric(df['turn'], errors='coerce')
                out_dict[code] = pd.Timestamp(out_date)
        except Exception:
            fail += 1
        done.append(code)

        n = len(done)
        if n % 100 == 0:
            el = time.time() - t0
            speed = (i + 1) / el if el > 0 else 0
            print(f"[进度] {n}/{len(delisted)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"close": close_dict, "turn": turn_dict,
                             "out": out_dict, "done": done}, f)

    print(f"[完成] 成功 {len(close_dict)} 只，失败 {fail} 只", flush=True)
    return close_dict, turn_dict, out_dict


lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

delisted = get_delisted()
print(f"2015 后退市股: {len(delisted)} 只", flush=True)

close_dict, turn_dict, out_dict = fetch(delisted)
close_df = pd.DataFrame(close_dict).sort_index()
turn_df = pd.DataFrame(turn_dict).sort_index()

with open(CACHE, 'wb') as f:
    pickle.dump({"close": close_df, "turn": turn_df, "outDate": out_dict}, f)
print(f"退市股缓存已保存 {CACHE}: {close_df.shape[0]}交易日 × {close_df.shape[1]}只", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

bs.logout()
