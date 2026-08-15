# -*- coding: utf-8 -*-
"""
抓取 618 只的涨跌停/停牌建模所需字段，带断点续跑
字段：open/high/low/preclose/tradestatus/pctChg/isST
用途：一字涨停(买不进)、一字跌停(卖不出)、停牌(无法交易) 建模
"""
import os
import pickle
import time
import pandas as pd
import baostock as bs

START, END = "2015-01-01", "2026-08-14"
CACHE = "limit_cache.pkl"
CHECKPOINT = "limit_checkpoint.pkl"
FIELDS = "date,open,high,low,preclose,tradestatus,pctChg,isST"

with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
CODES = list(listed['close'].columns)


def fetch(codes):
    data = {k: {} for k in ['open', 'high', 'low', 'preclose', 'tradestatus', 'pctChg', 'isST']}
    done = []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        data, done = cp['data'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in codes if c not in done]
    fail = 0
    t0 = time.time()
    for i, code in enumerate(todo):
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
                for k in data:
                    data[k][code] = pd.to_numeric(df[k], errors='coerce')
        except Exception:
            fail += 1
        done.append(code)
        n = len(done)
        if n % 100 == 0:
            el = time.time() - t0
            speed = (i + 1) / el if el > 0 else 0
            print(f"[进度] {n}/{len(codes)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"data": data, "done": done}, f)
    print(f"[完成] 成功 {len(data['open'])} 只，失败 {fail} 只", flush=True)
    return data


lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

print(f"待抓取涨跌停字段数据: {len(CODES)} 只", flush=True)
data = fetch(CODES)

result = {k: pd.DataFrame(data[k]).sort_index() for k in data}
with open(CACHE, 'wb') as f:
    pickle.dump(result, f)
print(f"缓存已保存 {CACHE}: {result['open'].shape[0]}交易日 × {result['open'].shape[1]}只", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
bs.logout()
