# -*- coding: utf-8 -*-
"""
抓取 618 只的市值代理（总股本 totalShare / 流通股本 liqaShare）+ 证监会行业分类
市值 = close × totalShare（快照式：取最新一期年报股本，股本变动慢、对横截面排序影响小）
"""
import os
import pickle
import time
import pandas as pd
import baostock as bs

CACHE = "size_industry_cache.pkl"
CHECKPOINT = "size_industry_checkpoint.pkl"

with open("full_market_cache.pkl", 'rb') as f:
    listed = pickle.load(f)
CODES = list(listed['close'].columns)


def fetch(codes):
    share_dict, ind_dict, done = {}, {}, []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        share_dict, ind_dict, done = cp['share'], cp['ind'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in codes if c not in done]
    fail = 0
    t0 = time.time()
    for i, code in enumerate(todo):
        # 股本：从最新年报往前找（2025 -> 2014），取第一个有数据的 totalShare
        ts = None
        for year in range(2025, 2013, -1):
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
        # 行业分类
        try:
            rs = bs.query_stock_industry(code=code)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                d = dict(zip(rs.fields, rows[0]))
                ind_dict[code] = d.get('industry', '')
            else:
                ind_dict[code] = ''
        except Exception:
            ind_dict[code] = ''
            fail += 1
        done.append(code)
        n = len(done)
        if n % 100 == 0:
            el = time.time() - t0
            speed = (i + 1) / el if el > 0 else 0
            print(f"[进度] {n}/{len(codes)}  失败{fail}  {speed:.2f}只/秒  已用{el:.0f}秒", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"share": share_dict, "ind": ind_dict, "done": done}, f)
    print(f"[完成] 股本成功 {sum(1 for v in share_dict.values() if v)} 只，失败 {fail} 只", flush=True)
    return share_dict, ind_dict


lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

print(f"待抓取市值/行业数据: {len(CODES)} 只", flush=True)
share, ind = fetch(CODES)

share_s = pd.Series(share)
ind_s = pd.Series(ind)

with open(CACHE, 'wb') as f:
    pickle.dump({"totalShare": share_s, "industry": ind_s}, f)
print(f"缓存已保存 {CACHE}: 股本 {share_s.notna().sum()}/618，行业 {ind_s.notna().sum()}/618", flush=True)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
bs.logout()
