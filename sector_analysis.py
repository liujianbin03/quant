# -*- coding: utf-8 -*-
"""
拉取新浪「电子信息」「电子器件」两个板块的成分股，按涨跌幅/成交额排序分析
用法：python sector_analysis.py
"""
import requests
import pandas as pd

# 新浪板块 node 代码（对应新浪行业的 label）
SECTORS = {
    "电子信息": "new_dzxx",
    "电子器件": "new_dzqj",
}

URL = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
       "Market_Center.getHQNodeData")


def fetch_sector(node, num=300):
    """拉取某个新浪板块 node 的成分股实时行情"""
    params = {
        "page": 1,
        "num": num,
        "sort": "changepercent",
        "asc": 0,          # 按涨跌幅降序
        "node": node,
    }
    r = requests.get(URL, params=params, timeout=15)
    r.encoding = "gbk"
    text = r.text
    # 新浪返回的是 JS 对象数组，不是标准 JSON，简单处理
    import json
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 处理带前缀的返回值，去掉非 JSON 前缀
        start = text.find("[")
        data = json.loads(text[start:])
    return pd.DataFrame(data)


for name, node in SECTORS.items():
    print(f"\n{'='*60}")
    print(f"板块：{name}  (node={node})")
    print(f"{'='*60}")
    try:
        df = fetch_sector(node)
        # 需要的字段
        cols = ["symbol", "name", "trade", "changepercent", "amount", "turnoverratio", "per"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        df["changepercent"] = df["changepercent"].astype(float)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") / 1e8  # 转亿
        df["trade"] = pd.to_numeric(df["trade"], errors="coerce")

        print(f"成分股数量：{len(df)}")

        # 领涨前10
        top = df.nlargest(10, "changepercent")[["symbol", "name", "trade", "changepercent", "amount"]]
        print("\n【涨幅前10】")
        print(top.to_string(index=False))

        # 成交额前10（资金关注度）
        amt = df.nlargest(10, "amount")[["symbol", "name", "trade", "changepercent", "amount"]]
        print("\n【成交额前10（资金流向）】")
        print(amt.to_string(index=False))

    except Exception as e:
        print(f"[失败] {name}: {type(e).__name__}: {e}")
