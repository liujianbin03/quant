# A股全市场量化因子研究

用 Python 对 A 股全市场系统抽样（每 8 取 1，618 只 × 2015–2026，11 年+）做因子挖掘、IC 检验、带成本回测、样本外验证与幸存者偏差修正的研究项目。

> 研究用途，非投资建议。回测结果含历史数据局限，不构成任何买卖依据。

## 核心结论（诚实版）

| 结论 | 说明 |
|------|------|
| **干净的价值 alpha 是 EP，不是 PB** | Barra 中性化后：PB 的 IC 从 +0.063 掉到 +0.040（大半是 size/行业暴露），EP 反从 +0.043 升到 +0.047（\|t\| 5.55 全场最高）——印证 Liu-Stambaugh-Yuan「中国市场 EP 优于 BM」 |
| **纯 PB 是"size+壳价值"假象** | 纯 PB 年化 +13.13% 大部分来自小盘暴露；退市修正后侵蚀 -7.26pp（踩雷 18 次）；此前 "-0.28pp" 是四因子合成口径，非纯 PB |
| **剔小30% 市值 = 消灭退市雷** | 终极检验（退市修正 × 剔小30%）：PB 踩雷 18→1，EP 5→0，PB+EP 4→0；三因子超额收益集体转正 |
| **PB+EP + 剔小30% 是最终胜者** | 超额 +3.55pp、踩雷 0 次、8 切点样本外 IC 0.073（100% 显著）、Barra 纯因子 IC 0.063 最高——唯一四重检验全过 |
| **ML 无新增 alpha** | LightGBM 10 特征 walk-forward：全样本 +37% 是"小盘反转"幻觉，剔小30%后超额归零（+0.06pp）；特征重要性显示模型只学动量/反转/波动，估值垫底 |
| **A 股是反转不是动量** | 动量 IC -0.054（\|t\| 3.54），反转 IC -0.073（\|t\| 5.06）；但反转收益 63% 是幸存者偏差虚增 |
| **质量因子不提供增量** | ROE/毛利率/净利增速/营收增速/负债率叠到价值上全部稀释 alpha |
| **幸存者偏差是反转策略头号杀手** | 反转年化被虚增 ~4.8pp（+7.64% → +2.85%）；面值过滤(<1元)可避 90% 雷 |
| **低换手 = 低摩擦 = 可执行** | 价值策略换手 ~10%/期，vs 反转 55%/期 |

一句话：**中国市场的干净价值 alpha 是 EP（盈利收益率）；PB 的强势是 size/壳价值假象；「PB+EP 且剔除最小 30% 市值」是唯一经得起「退市修正 + 多切点样本外 + Barra 中性化 + 成本」四重检验的策略；ML、反转、质量都被成本或幸存者偏差吃掉。**

## 目录结构

```
quant/
├── config.py              # 路径与运行约定（所有脚本在根目录运行）
├── fetch/                 # 数据抓取（14个，含限速+断点续跑）
│   ├── full_market_ic.py  # 全市场 close/turn（主价格源）
│   ├── fetch_valuation.py # PE/PB/PS/PCF 估值
│   ├── fetch_delisted*.py # 退市股 价格/估值/股本
│   ├── fetch_roe.py / fetch_quality.py  # 质量因子（akshare 新浪）
│   ├── fetch_size_industry.py  # 股本/行业
│   └── fetch_index_name.py      # 沪深300 + 股票名称
├── research/              # 因子研究/回测/验证（45个，历史记录）
│   ├── value_factor.py / ep_factor.py / value_ep_*.py  # 价值/EP 研究主线
│   ├── reaction_factors.py   # 市场反应因子
│   ├── ml_cross_section.py   # LightGBM 截面（结论：无新增 alpha）
│   ├── goal_search.py / goal_sim.py / risk_*.py / delever_*.py  # 目标迭代/风控证伪
│   └── multiple_testing.py / cost_sensitivity.py  # 方法论检验
├── practice/              # 实盘工具（5个）
│   ├── signal_picker.py      # 选股清单
│   ├── paper_trade.py        # 纸面跟踪
│   ├── performance_report.py # 绩效+归因报告
│   ├── refresh_data.py       # 月度数据刷新
│   └── check_baostock.py     # baostock 解封检测
├── tests/                 # 连接与功能测试
├── figures/               # 结果图表（本地生成，不入库）
├── reports/               # 绩效报告（markdown）
└── *.pkl / *.csv          # 数据缓存（本地，不入库）
```

## 数据源

- **baostock** — 全市场日线（含退市股，前复权 `adjustflag="2"`），免费，~0.47 只/秒
  > ⚠️ 所有 baostock 抓取脚本已内置 0.5s/请求限速、并发降至 2。高频/并发会触发 IP 黑名单；若被封，换 IP（重启路由器/手机热点/VPN）后跑 `practice/check_baostock.py` 即可恢复。
- **akshare** — 新浪财务接口 `stock_financial_analysis_indicator`（ROE/毛利率/负债率等）
- **新浪** `stock_zh_a_daily` — 价格数据备用源（东财 `stock_zh_a_hist` 不稳定）

## 环境与依赖

- Python 3.10+（开发环境 3.10.9）
- `akshare` `baostock` `backtrader` `pandas` `numpy` `matplotlib`
- `scipy` `scikit-learn` `lightgbm`（仅 `ml_cross_section.py` 需要）

```bash
pip install akshare baostock backtrader pandas numpy matplotlib scipy scikit-learn lightgbm
```

## 运行方式

> 所有脚本在**项目根目录**运行（相对路径按根目录解析）。

1. 先跑 `fetch/` 下脚本生成数据缓存（`.pkl`，已 gitignore，需本地重新抓取）
2. 再跑 `research/` 下分析脚本出 IC 表 + 回测结果 + PNG
3. 实盘流程见 `STRATEGY.md`：`python practice/refresh_data.py` → `practice/paper_trade.py` → `practice/performance_report.py`

> 数据缓存文件（`*.pkl` / `*.csv`）不随仓库分发，需用 `fetch/` 下脚本重新抓取。抓取耗时：全市场价格 ~15 分钟，质量因子 ~13 分钟。

## 回测参数约定

- 月度调仓，HOLD=21 天，N_HOLD=50（top50 等权）
- 成本：ROUND_TRIP=0.003（佣金万2.5 + 印花税万5 + 滑点千1）
- 区间：2015–2026
- 估值因子 4 个月年报发布滞后，避免前视偏差

## 免责声明

本项目仅用于量化研究方法学习与验证，不构成任何投资建议。历史回测不等于未来收益，市场有风险，投资需谨慎。
