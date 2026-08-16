# -*- coding: utf-8 -*-
"""
HTML 看板：生成 reports/dashboard.html（自包含、双击浏览器即可看，无需服务器）
内容：绩效指标卡 + 当前 top20 持仓 + 目标对标 + 因子归因
用法（项目根目录）：python practice/dashboard.py
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd

LISTED_CACHE = "full_market_cache.pkl"
VAL_CACHE = "val_cache.pkl"
SIZE_CACHE = "size_industry_cache.pkl"
INDEX_NAME_CACHE = "index_name_cache.pkl"

HOLD, N_HOLD = 21, 20
ROUND_TRIP = 0.0030
SMALL_PCT = 0.30
WARMUP = 126


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def clean(f):
    f = f.astype(float)
    f = f.where(f > 0)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return f.clip(lo, hi)


def winsorize(s, q=0.01):
    s = s.astype(float)
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


with open(LISTED_CACHE, 'rb') as f:
    listed = pickle.load(f)
with open(VAL_CACHE, 'rb') as f:
    val = pickle.load(f)
with open(SIZE_CACHE, 'rb') as f:
    si = pickle.load(f)
with open(INDEX_NAME_CACHE, 'rb') as f:
    iname = pickle.load(f)

close = listed['close']
turn = listed['turn'].reindex(close.index)
pe = val['pe'].reindex(close.index)
pb = val['pb'].reindex(close.index)
total_share = si['totalShare'].reindex(close.columns)
names = iname['name'].reindex(close.columns)
hs300 = iname['index'].reindex(close.index)
daily_ret = close.pct_change(fill_method=None)

with np.errstate(divide='ignore', invalid='ignore'):
    ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)

mcap = close.mul(total_share, axis=1)
dates = close.index
rebal = [i for i in range(WARMUP, len(dates), HOLD) if i + HOLD < len(dates)]


def score_strategy(i):
    pbep = -zscore(clean(pb.iloc[i])) + zscore(winsorize(ep.iloc[i]))
    t = -zscore(winsorize(turn.iloc[i - 21:i].mean()))
    lc = -zscore(winsorize((daily_ret.iloc[i - 63:i] >= 0.095).sum()))
    return pbep + t + lc


def universe_ex_small(i):
    mi = mcap.iloc[i]
    return mi[mi > mi.quantile(SMALL_PCT)].index


# 当前 top20
i = len(dates) - 1
sc = score_strategy(i)
valid = sc.dropna().index.intersection(universe_ex_small(i))
top = sc[valid].nlargest(N_HOLD)

# 回测绩效
rets, prev = [], set()
for k in rebal:
    sc = score_strategy(k)
    fwd = close.iloc[k + HOLD] / close.iloc[k] - 1
    v = sc.dropna().index.intersection(universe_ex_small(k))
    if len(v) >= N_HOLD:
        tp = sc[v].nlargest(N_HOLD).index
        turnv = 1 - len(set(tp) & prev) / N_HOLD if prev else 1.0
        rets.append(fwd[tp].mean() - turnv * ROUND_TRIP)
        prev = set(tp)
s = pd.Series(rets)
cum = (1 + s).cumprod()
annual = (cum.iloc[-1] ** (12 / len(s)) - 1) * 100
sharpe = s.mean() / s.std() * np.sqrt(12)
mdd = (cum / cum.cummax() - 1).min() * 100
hs_cum = hs300.iloc[len(dates) - 1] / hs300.iloc[rebal[0]] - 1

# 纸面跟踪状态
paper = None
if os.path.exists("paper_trade_state.pkl"):
    with open("paper_trade_state.pkl", 'rb') as f:
        paper = pickle.load(f)

# 归因
uni = valid
rows_attr = [
    ("PB", pb.iloc[i][top.index].median(), pb.iloc[i][uni].median()),
    ("EP%", ep.iloc[i][top.index].median() * 100, ep.iloc[i][uni].median() * 100),
    ("换手率", turn.iloc[i - 21:i].mean()[top.index].median(), turn.iloc[i - 21:i].mean()[uni].median()),
    ("涨停次数", (daily_ret.iloc[i - 63:i] >= 0.095).sum()[top.index].mean(), (daily_ret.iloc[i - 63:i] >= 0.095).sum()[uni].mean()),
]

# 目标对标
t1 = "✓" if cum.iloc[-1] >= 1 else "✗"
t2 = "✓" if mdd >= -15 else "✗"
t3 = "✓" if sharpe >= 0.4 else "✗"

# 生成 HTML
holding_rows = ""
for rank, code in enumerate(top.index, 1):
    nm = names[code] if pd.notna(names[code]) else '?'
    holding_rows += (f"<tr><td>{rank}</td><td>{code}</td><td>{nm}</td>"
                     f"<td>{close.iloc[i][code]:.2f}</td><td>{pb.iloc[i][code]:.2f}</td>"
                     f"<td>{ep.iloc[i][code]*100:.2f}%</td></tr>")

attr_rows = ""
for name, a, b in rows_attr:
    attr_rows += f"<tr><td>{name}</td><td>{a:.2f}</td><td>{b:.2f}</td></tr>"

paper_html = ""
if paper:
    paper_html = (f"<p>纸面跟踪：起始 {paper['date'].date()}，当前净值 {paper['nav']:.4f}，"
                  f"沪深300起始 {paper.get('nav0_index', paper['index']):.0f}</p>")

html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>A股价值策略看板</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:20px;color:#222}}
h1{{font-size:20px}} .cards{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{border:1px solid #ddd;border-radius:8px;padding:14px 18px;min-width:140px}}
.card b{{font-size:22px;display:block}} .ok{{color:#1a7f37}} .no{{color:#c62828}}
table{{border-collapse:collapse;margin:10px 0;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#f5f5f5}} h2{{font-size:16px;margin-top:24px;border-bottom:2px solid #333;padding-bottom:4px}}
.muted{{color:#888;font-size:13px}}
</style></head><body>
<h1>A股价值策略看板 <span class="muted">PB+EP+换手率+涨停次数 · top20 · 剔小30%</span></h1>
<p class="muted">数据截至 {dates[i].date()} · 生成于 {datetime.date.today()}</p>
{paper_html}
<h2>绩效（2015–2026 回测，含成本）</h2>
<div class="cards">
<div class="card">年化收益<b>{annual:+.2f}%</b></div>
<div class="card">超额夏普<b>{sharpe:.2f}</b></div>
<div class="card">最大回撤<b>{mdd:.2f}%</b></div>
<div class="card">累计净值<b>{cum.iloc[-1]:.2f}</b></div>
<div class="card">沪深300累计<b>{hs_cum*100:+.1f}%</b></div>
</div>
<h2>目标对标</h2>
<div class="cards">
<div class="card">跑赢沪深300<b class="{'ok' if t1=='✓' else 'no'}">{t1}</b></div>
<div class="card">回撤≤15%<b class="{'ok' if t2=='✓' else 'no'}">{t2}</b></div>
<div class="card">夏普≥0.4<b class="{'ok' if t3=='✓' else 'no'}">{t3}</b></div>
</div>
<h2>当前持仓 top{N_HOLD}</h2>
<table><tr><th>#</th><th>代码</th><th>名称</th><th>现价</th><th>PB</th><th>EP%</th></tr>{holding_rows}</table>
<h2>因子归因（持仓 vs 候选池）</h2>
<table><tr><th>因子</th><th>持仓均值</th><th>候选池均值</th></tr>{attr_rows}</table>
<p class="muted">研究用途，非投资建议。历史回测≠未来收益。</p>
</body></html>"""

os.makedirs("reports", exist_ok=True)
with open("reports/dashboard.html", 'w', encoding='utf-8') as f:
    f.write(html)
print(f"[OK] 看板已保存 reports/dashboard.html")
