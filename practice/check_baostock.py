# -*- coding: utf-8 -*-
"""
换 IP 后：一键检测 baostock 是否解封，解封则自动补抓退市股股本。

用法（在你换完 IP —— 重启路由器/连手机热点/挂 VPN 之后）：
    python check_baostock.py

行为：
  1. 检测 baostock 登录，打印当前出口 IP 归属
  2. 若已解封 → 自动执行 fetch_delisted_share.py 补抓退市股股本
  3. 若仍黑名单 → 提示换个网络再试（重启路由器 / 手机热点 / VPN）
"""
import sys
import subprocess
import baostock as bs

lg = bs.login()
if lg.error_code == "0":
    print(f"[OK] baostock 已解封，可正常抓数据。开始补抓退市股股本...\n")
    bs.logout()
    # 调用抓取脚本（已内置限速 0.5s/请求）
    r = subprocess.run([sys.executable, "fetch/fetch_delisted_share.py"])
    sys.exit(r.returncode)
else:
    print(f"[黑名单未解除] code={lg.error_code} msg={lg.error_msg}")
    print("\n请换个网络出口再试，任选其一：")
    print("  1) 重启家用路由器（断电30秒再开，动态IP重拨后大概率换新IP）")
    print("  2) 电脑连手机热点（出口变成手机运营商IP）")
    print("  3) 挂 VPN")
    print("\n换完后再跑一次：python check_baostock.py")
    bs.logout()
    sys.exit(1)
