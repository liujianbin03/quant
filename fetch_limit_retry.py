# -*- coding: utf-8 -*-
"""补抓失败缺失的涨跌停字段（低并发4进程），合并进 limit_cache.pkl"""
import json
import os
import pickle
import time
import pandas as pd
import baostock as bs
from multiprocessing import Pool

START, END = "2015-01-01", "2026-08-14"
CACHE = "limit_cache.pkl"
FIELDS = "date,open,high,low,preclose,tradestatus,pctChg,isST"
WORKERS = 2  # 降并发限速：避免被 baostock 封 IP

with open("limit_missing.json") as f:
    CODES = json.load(f)


def fetch_one(code):
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
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    print(f"补抓 {len(CODES)} 只失败股，{WORKERS} 进程", flush=True)
    fail = 0
    with Pool(WORKERS) as pool:
        for j, (code, out) in enumerate(pool.imap_unordered(fetch_one, CODES), 1):
            if out is not None:
                for k in data:
                    data[k][code] = out[k]
            else:
                fail += 1
            if j % 20 == 0:
                print(f"[进度] {j}/{len(CODES)}  失败{fail}", flush=True)
    print(f"[完成] 失败 {fail} 只", flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    total = data['open'].shape[1]
    print(f"缓存已更新: {data['open'].shape[0]}交易日 × {total}只", flush=True)


if __name__ == '__main__':
    main()
