# -*- coding: utf-8 -*-
"""测试新浪财务指标接口"""
import akshare as ak

try:
    df = ak.stock_financial_analysis_indicator(symbol="600519", start_year="2023")
    print("成功！返回列名：")
    for c in df.columns:
        print(f"  - {c}")
    print("\n前3行（关键列）：")
    keys = [c for c in ["日期", "净资产收益率(%)", "销售毛利率(%)", "资产负债率(%)",
                        "净利润增长率(%)", "主营业务收入增长率(%)"] if c in df.columns]
    print(df[keys].head(3).to_string(index=False))
except Exception as e:
    print(f"[失败] {type(e).__name__}: {e}")
