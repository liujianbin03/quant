# -*- coding: utf-8 -*-
"""单线程补抓剩余失败股（避免并发连接被服务器拒绝），合并进 limit_cache.pkl"""
import json
import pickle
import time
import pandas as pd
import baostock as bs

START, END = "2015-01-01", "2026-08-14"
CACHE = "limit_cache.pkl"
FIELDS = "date,open,high,low,preclose,tradestatus,pctChg,isST"

with open("limit_missing.json") as f:
    CODES = json.load(f)


def main():
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    lg = bs.login()
    print(f"单线程补抓 {len(CODES)} 只", flush=True)
    fail = 0
    t0 = time.time()
    for j, code in enumerate(CODES, 1):
        time.sleep(0.5)  # 限速：避免高频请求被 baostock 封 IP
        try:
            rs = bs.query_history_k_data_plus(code, FIELDS,
                start_date=START, end_date=END, frequency="d", adjustflag="2")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if len(rows) >= 20:
                df = pd.DataFrame(rows, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                for k in ['open', 'high', 'low', 'preclose', 'tradestatus', 'pctChg', 'isST']:
                    data[k][code] = pd.to_numeric(df[k], errors='coerce')
            else:
                fail += 1
        except Exception:
            fail += 1
        if j % 10 == 0:
            el = time.time() - t0
            print(f"[进度] {j}/{len(CODES)}  失败{fail}  {j/el:.2f}只/秒", flush=True)
    bs.logout()
    print(f"[完成] 失败 {fail} 只", flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    print(f"缓存已更新: {data['open'].shape[0]}交易日 × {data['open'].shape[1]}只", flush=True)


if __name__ == '__main__':
    main()
