# -*- coding: utf-8 -*-
"""
拉取 254 只退市股的 totalShare（退市前最后一期年报股本），带断点续跑

用途：补齐「退市股修正 + 剔小30%市值」组合检验所缺的退市股市值数据。
      退市股当前没有 totalShare，无法计算市值、无法参与"剔除最小30%"排序。

⚠️ 本脚本用 baostock。若在沙箱/云端环境运行报「黑名单用户」，
   请在你自己本机（baostock 可用处）运行：
       python fetch_delisted_share.py
   跑完会生成 delisted_share_cache.pkl（约 254 只，几分钟）。

逻辑：从退市年份往前逐年找，取第一个有 totalShare 的年报（quarter=4）。
"""
import os
import pickle
import time
import pandas as pd
import baostock as bs

CACHE = "delisted_share_cache.pkl"
CHECKPOINT = "delisted_share_checkpoint.pkl"

with open("delisted_cache.pkl", 'rb') as f:
    delisted = pickle.load(f)
CODES = list(delisted['close'].columns)
OUT_DATE = delisted.get('outDate', {})


def fetch(codes):
    share_dict, done = {}, []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        share_dict, done = cp['share'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in codes if c not in done]
    fail = 0
    t0 = time.time()
    for i, code in enumerate(todo):
        time.sleep(0.5)  # 限速：避免高频请求被 baostock 封 IP
        out_year = int(OUT_DATE.get(code, pd.Timestamp('2020-01-01')).year)
        ts = None
        for year in range(out_year, 2013, -1):   # 从退市年往前找
            try:
                rs = bs.query_profit_data(code, year, 4)
                while rs.next():
                    r = rs.get_row_data()
                    d = dict(zip(rs.fields, r))
                    v = d.get('totalShare')
                    if v not in (None, '', '0'):
                        ts = float(v)
                        break
                if ts:
                    break
            except Exception:
                pass
        share_dict[code] = ts
        done.append(code)
        n = len(done)
        if n % 50 == 0:
            el = time.time() - t0
            speed = (i + 1) / el if el > 0 else 0
            print(f"[进度] {n}/{len(codes)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"share": share_dict, "done": done}, f)

    print(f"[完成] 有股本 {sum(1 for v in share_dict.values() if v)} 只，失败 {fail} 只", flush=True)
    return share_dict


lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}（若为黑名单，请在本机运行本脚本）")
    raise SystemExit

print(f"待抓取退市股股本: {len(CODES)} 只", flush=True)
share = fetch(CODES)

share_s = pd.Series(share)
with open(CACHE, 'wb') as f:
    pickle.dump(share_s, f)
print(f"缓存已保存 {CACHE}: 有股本 {share_s.notna().sum()}/{len(CODES)}", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
bs.logout()
