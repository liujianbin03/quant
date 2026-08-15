# -*- coding: utf-8 -*-
"""
月度数据刷新：把 fetch 脚本的 END 更新为指定日期（默认今天），删除价格缓存后按序重抓。
用法：
    python refresh_data.py            # 刷新到今天
    python refresh_data.py 2026-09-30 # 刷新到指定日期

说明：
  - 前复权(adjustflag=2)价格会随新分红/送股回溯调整，故必须「全量重抓」而非增量追加，
    否则新旧数据口径不一致、会算错收益。
  - 全程已限速(0.5s/请求)，约 15-20 分钟，带断点续跑，可随时中断重跑。
  - 刷新完成后：python paper_trade.py 记录本月 → python performance_report.py 出报告。
"""
import os
import re
import sys
import subprocess
import datetime

end = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()

# (脚本, 说明, 是否删除其缓存以强制重抓)
STEPS = [
    ("full_market_ic.py",   "close/turn 价格(前复权)", "full_market_cache.pkl"),
    ("fetch_valuation.py",  "PE/PB/PS/PCF 估值",      "val_cache.pkl"),
    ("fetch_index_name.py", "沪深300 + 股票名称",      None),
]

print(f"目标刷新日期: {end}\n")

# 1) 更新各脚本的 END
for fn, desc, _ in STEPS:
    with open(fn, encoding='utf-8') as f:
        txt = f.read()
    new_txt = re.sub(r'(START,\s*END\s*=\s*"[^"]*",\s*)"[^"]*"',
                     lambda m: m.group(1) + f'"{end}"', txt)
    if new_txt == txt:
        print(f"[跳过] {fn}: 未找到 END 行（格式不符）")
    else:
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(new_txt)
        print(f"[更新] {fn}: END -> {end}")

# 2) 删除价格/估值缓存以强制全量重抓（前复权要求）
for fn, desc, cache in STEPS:
    if cache and os.path.exists(cache):
        os.remove(cache)
        print(f"[删除缓存] {cache}（强制全量重抓 {desc}）")

print("\n开始抓取（约 15-20 分钟，已限速）...\n")
for fn, desc, _ in STEPS:
    print(f"=== {desc}  ({fn}) ===", flush=True)
    r = subprocess.run([sys.executable, fn])
    if r.returncode != 0:
        print(f"  [失败] {fn} 退出码 {r.returncode}，中断。可重跑本脚本续跑。", flush=True)
        sys.exit(1)

print("\n刷新完成。下一步：")
print("  1) python paper_trade.py       # 记录本月前向收益、滚动净值")
print("  2) python performance_report.py # 出绩效与归因报告")
