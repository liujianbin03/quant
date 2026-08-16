# -*- coding: utf-8 -*-
"""
全 A 股 + 长周期 因子 IC 分析（断点续跑版）
- 股票池：全 A 股系统抽样（每8只取1），剔除 ST/退市/B股/北交所
- 周期：2015-01 ~ 2026-08（约11年）
- 因子：动量(6月) / 短期反转(1月) / 波动率(6月) / 换手率(1月均值)
- 方法：月度换仓，Spearman 秩 IC，汇总 ICIR / t值 / 胜率
用法：python -u full_market_ic.py
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import baostock as bs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

START, END = "2015-01-01", "2026-08-14"
LOOKBACK = 126   # 6个月（动量/波动窗口）
REV = 21         # 1个月（反转窗口）
HOLD = 21        # 持有1个月
CACHE = "full_market_cache.pkl"
CHECKPOINT = "full_market_checkpoint.pkl"
SAMPLE_STEP = 8  # 系统抽样：每8只取1（~625只，兼顾速度与统计功效）

# 因子：(名称, 期望方向) +1=越高越好  -1=越低越好
FACTORS = [
    ("动量(6月)", +1),
    ("反转(1月)", -1),
    ("波动率(6月)", -1),
    ("换手率(1月)", -1),
]


def get_universe():
    rs = bs.query_stock_basic()
    codes = []
    while rs.next():
        row = rs.get_row_data()
        code, name, ipo, out, typ, status = row
        if typ != "1" or status != "1":
            continue
        if not (code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3")):
            continue
        if "ST" in name or "退" in name:
            continue
        codes.append(code)
    return codes


def fetch_all(codes):
    """断点续跑：把 dict 存进 checkpoint，失败/中断可恢复"""
    close_dict, turn_dict, done = {}, {}, []
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            cp = pickle.load(f)
        close_dict, turn_dict, done = cp['close'], cp['turn'], cp['done']
        print(f"[续跑] 已恢复 {len(done)} 只", flush=True)

    todo = [c for c in codes if c not in done]
    fail = 0
    t_start = time.time()
    for i, code in enumerate(todo):
        try:
            rs = bs.query_history_k_data_plus(
                code, "date,close,turn",
                start_date=START, end_date=END,
                frequency="d", adjustflag="2")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if len(rows) >= LOOKBACK + HOLD + 5:
                df = pd.DataFrame(rows, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                close_dict[code] = pd.to_numeric(df['close'], errors='coerce')
                turn_dict[code] = pd.to_numeric(df['turn'], errors='coerce')
        except Exception:
            fail += 1
        done.append(code)

        n = len(done)
        if n % 150 == 0:
            el = time.time() - t_start
            speed = (i + 1) / el if el > 0 else 0
            eta = (len(todo) - i - 1) / speed / 60 if speed > 0 else 0
            print(f"[进度] {n}/{len(codes)}  失败{fail}  "
                  f"{speed:.2f}只/秒  预计剩余{eta:.1f}分钟", flush=True)
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({"close": close_dict, "turn": turn_dict, "done": done}, f)

    print(f"[完成] 成功 {len(close_dict)} 只，失败 {fail} 只", flush=True)
    return close_dict, turn_dict


def spearman(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 10:
        return np.nan
    return df['a'].rank().corr(df['b'].rank())


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


# ============ 数据 ============
lg = bs.login()
if lg.error_code != "0":
    print(f"登录失败: {lg.error_msg}")
    raise SystemExit

all_codes = get_universe()
codes = all_codes[::SAMPLE_STEP]
print(f"全 A 股共 {len(all_codes)} 只，系统抽样(每{SAMPLE_STEP}取1)后 {len(codes)} 只", flush=True)

if os.path.exists(CACHE):
    print("发现完整缓存，直接加载...", flush=True)
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    close_df, turn_df = data['close'], data['turn']
else:
    print("开始拉取日线（断点续跑，可随时中断重跑）...", flush=True)
    close_dict, turn_dict = fetch_all(codes)
    close_df = pd.DataFrame(close_dict).sort_index()
    turn_df = pd.DataFrame(turn_dict).sort_index()
    with open(CACHE, 'wb') as f:
        pickle.dump({"close": close_df, "turn": turn_df}, f)
    print(f"完整缓存已保存 {CACHE}", flush=True)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

bs.logout()

print(f"价格矩阵：{close_df.shape[0]} 交易日 × {close_df.shape[1]} 只股票", flush=True)
daily_ret = close_df.pct_change(fill_method=None)
dates = close_df.index

# ============ 逐期 IC ============
ic_collect = {n: [] for n, _ in FACTORS}
pool = {n: [] for n, _ in FACTORS}

i = LOOKBACK
periods = 0
while i + HOLD < len(dates):
    t0 = dates[i - LOOKBACK]
    t1 = dates[i]
    t_rev = dates[i - REV]
    t2 = dates[i + HOLD]

    mom = close_df.loc[t1] / close_df.loc[t0] - 1
    rev = close_df.loc[t1] / close_df.loc[t_rev] - 1
    vol = daily_ret.loc[t0:t1].std()
    turn = turn_df.loc[t_rev:t1].mean()
    fwd = close_df.loc[t2] / close_df.loc[t1] - 1

    fmap = {"动量(6月)": mom, "反转(1月)": rev,
            "波动率(6月)": vol, "换手率(1月)": turn}
    for name, _ in FACTORS:
        f = fmap[name]
        ic = spearman(f, fwd)
        if not np.isnan(ic):
            ic_collect[name].append(ic)
        pool[name].append(zscore(f))

    i += HOLD
    periods += 1

print(f"\n共 {periods} 个换仓期\n", flush=True)

# ============ IC 汇总 ============
print("=" * 84)
print(f"全 A 股因子 IC 分析（{close_df.shape[1]} 只股票，{periods} 期，2015-2026）")
print("=" * 84)
print(f"{'因子':<14}{'IC均值':>9}{'IC标准差':>9}{'ICIR':>9}{'|t值|':>9}{'胜率':>8}")
print("-" * 84)

for name, direction in FACTORS:
    s = pd.Series(ic_collect[name]).dropna()
    ic_mean = s.mean()
    ic_std = s.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    t = abs(icir) * np.sqrt(len(s))
    win = ((s * direction) > 0).mean() * 100
    mark = "  <== 显著" if t >= 2 else ""
    print(f"{name:<14}{ic_mean:>9.4f}{ic_std:>9.4f}{icir:>9.3f}{t:>9.2f}{win:>7.0f}%{mark}", flush=True)

print("-" * 84)
print("判断标准：|t|>2 统计显著；ICIR>0.3 好因子")
print("方向：反转/波动率/换手率 期望 IC 为负（过去涨多→未来跌、低波好、低换手好）", flush=True)

# ============ 相关性矩阵 ============
names = [n for n, _ in FACTORS]
corr_df = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        ca = pd.concat(pool[a]).reset_index(drop=True)
        cb = pd.concat(pool[b]).reset_index(drop=True)
        corr_df.loc[a, b] = ca.corr(cb)
print("\n因子相关性矩阵：")
print(corr_df.round(3).to_string())

# ============ 图1：IC 时序 ============
plt.figure(figsize=(14, 6))
for name, _ in FACTORS:
    plt.plot(ic_collect[name], marker='o', markersize=2, linewidth=1, label=name, alpha=0.8)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title(f"全 A 股各因子 IC 时序（{periods} 期，2015-2026）")
plt.xlabel("换仓期")
plt.ylabel("IC (秩相关)")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/full_market_ic_ts.png", dpi=150)
print("\n[OK] IC时序图已保存 figures/full_market_ic_ts.png", flush=True)

# ============ 图2：相关性热力图 ============
plt.figure(figsize=(8, 7))
mat = corr_df.values.astype(float)
plt.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1)
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        plt.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', fontsize=10, color='black')
plt.colorbar(label='相关系数')
plt.title("全 A 股因子相关性矩阵")
plt.tight_layout()
plt.savefig("figures/full_market_corr.png", dpi=150)
print("[OK] 相关性热力图已保存 figures/full_market_corr.png", flush=True)
