# -*- coding: utf-8 -*-
"""
一键月度流水线：刷新数据 → 纸面跟踪 → 绩效报告 → HTML看板
用法（项目根目录）：python practice/run_monthly.py
可配合 Windows 任务计划程序每月定时运行。
"""
import subprocess
import sys

STEPS = [
    ("practice/refresh_data.py",       "刷新数据（抓价格/估值/指数，约15-20分钟）"),
    ("practice/paper_trade.py",        "记录本月前向收益、滚动净值"),
    ("practice/performance_report.py", "生成绩效+归因报告(markdown)"),
    ("practice/dashboard.py",          "生成 HTML 看板"),
]

for fn, desc in STEPS:
    print(f"\n=== {desc} ({fn}) ===", flush=True)
    r = subprocess.run([sys.executable, fn])
    if r.returncode != 0:
        print(f"[中断] {fn} 退出码 {r.returncode}", flush=True)
        sys.exit(1)

print("\n[完成] 月度流水线全部跑完。看板：reports/dashboard.html")
