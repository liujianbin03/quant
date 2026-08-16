# -*- coding: utf-8 -*-
"""
并行抓取 618 只涨跌停字段（多进程），复用断点 limit_checkpoint.pkl
字段：open/high/low/preclose/tradestatus/pctChg/isST
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import baostock as bs
from multiprocessing import Pool

START, END = "2015-01-01", "2026-08-14"
CACHE = "limit_cache.pkl"
CHECKPOINT = "limit_checkpoint.pkl"
FIELDS = "date,open,high,low,preclose,tradestatus,pctChg,isST"
WORKERS = 2  # 降并发限速：8 进程并发会触发 baostock 封 IP（黑名单）

with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
CODES = list(listed['close'].columns)


def fetch_one(code):
    """单只抓取，返回 (code, dict_of_series 或 None)"""
    time.sleep(0.5)  # 限速：避免并发高频请求被封 IP
    try:
        lg = bs.login()
        rs = bs.query_history_k_data_plus(code, FIELDS,
            start_date=START, end_date=END, frequency="d", adjustflag="2")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if len(rows) < 20:
            return code, None
        df = pd.DataFrame(rows, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        out = {k: pd.to_numeric(df[k], errors='coerce') for k in
               ['open', 'high', 'low', 'preclose', 'tradestatus', 'pctChg', 'isST']}
        return code, out
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return code, None


def main():
    data = {k: {} for k in ['open', 'high', 'low', 'preclose', 'tradestatus', 'pctChg', 'isST']}
    done = []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        data, done = cp['data'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in CODES if c not in done]
    print(f"待抓取 {len(todo)} 只，{WORKERS} 进程并行", flush=True)
    t0 = time.time()
    fail = 0
    with Pool(WORKERS) as pool:
        for j, (code, out) in enumerate(pool.imap_unordered(fetch_one, todo), 1):
            if out is not None:
                for k in data:
                    data[k][code] = out[k]
            else:
                fail += 1
            done.append(code)
            n = len(done)
            if n % 50 == 0:
                el = time.time() - t0
                speed = n / el
                print(f"[进度] {n}/{len(CODES)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
                with open(CHECKPOINT, 'wb') as f:
                    pickle.dump({"data": data, "done": done}, f)

    print(f"[完成] 成功 {len(data['open'])} 只，失败 {fail} 只", flush=True)
    result = {k: pd.DataFrame(data[k]).sort_index() for k in data}
    with open(CACHE, 'wb') as f:
        pickle.dump(result, f)
    print(f"缓存已保存 {CACHE}: {result['open'].shape[0]}交易日 × {result['open'].shape[1]}只", flush=True)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)


if __name__ == '__main__':
    main()
