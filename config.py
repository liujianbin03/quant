# -*- coding: utf-8 -*-
"""
项目路径与约定（config）

目录结构：
    quant/
    ├── fetch/       # 数据抓取（baostock / akshare / 新浪）
    ├── research/    # 因子研究、回测、验证（历史记录）
    ├── practice/    # 实盘工具（选股 / 纸面跟踪 / 报告 / 刷新）
    ├── tests/       # 连接与功能测试
    ├── figures/     # 结果图表（本地生成，不入库）
    ├── reports/     # 绩效与归因报告（持久化 markdown）
    └── *.pkl/*.csv  # 数据缓存（本地，已 gitignore）

运行约定：所有脚本在【项目根目录】运行，例如：
    python fetch/fetch_valuation.py
    python practice/signal_picker.py 20
    python practice/refresh_data.py

脚本内引用数据缓存用相对文件名（如 "full_market_cache.pkl"），
因为运行 CWD 是根目录；输出到 figures/ 或 reports/。
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(ROOT, "figures")
REPORTS = os.path.join(ROOT, "reports")
