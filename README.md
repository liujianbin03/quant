# A股全市场量化因子研究

用 Python 对 A 股全市场（618 只 × 2015–2026，11 年+）做因子挖掘、IC 检验、带成本回测、样本外验证与幸存者偏差修正的研究项目。

> 研究用途，非投资建议。回测结果含历史数据局限，不构成任何买卖依据。

## 核心结论（诚实版）

| 结论 | 说明 |
|------|------|
| **PB 价值因子是真 alpha** | IC +0.059（\|t\| 3.63），年化 +13.56%，换手仅 17%/期，幸存者偏差侵蚀仅 -0.28pp |
| **A 股是反转不是动量** | 动量 IC -0.054（\|t\| 3.54），反转 IC -0.073（\|t\| 5.06） |
| **质量因子不提供增量** | ROE/毛利率/净利增速/营收增速/负债率叠到 PB 上全部稀释 alpha，纯 PB 最优 |
| **幸存者偏差是反转策略头号杀手** | 反转策略年化被幸存者偏差虚增 ~4.8pp（+7.64% → +2.85%），63% 收益是假的 |
| **面值过滤能救命** | 剔除 <1 元股可避免 90% 踩雷（退市） |
| **低换手 = 低摩擦 = 可执行** | 价值策略换手 17%/期，vs 反转 55%/期 |

一句话：**低 PB 的价值策略（低换手、低退市风险、样本外稳定）是这套数据里唯一站得住的 alpha；反转/质量都被成本、幸存者偏差或噪声吃掉。**

## 目录结构

```
quant/
├── fetch_*.py            # 数据抓取（baostock / akshare / 新浪），带断点续跑
├── value_factor.py       # PB/PE/PS/PCF 估值因子 IC + 回测
├── value_oos.py          # 价值因子样本外验证（2021-01 切分）
├── value_survivorship.py # 价值策略幸存者偏差检验
├── value_quality.py      # PB + ROE 双因子（价值+质量，方案 A）
├── value_quality2.py     # PB + 5 质量因子增量 alpha 检验
├── full_market_ic.py     # 全市场动量/反转/波动/换手 IC 分析
├── cost_backtest.py      # 带成本回测（佣金+印花税+滑点）
├── oos_validation.py     # 反转策略样本外验证
├── survivorship_bias.py  # 幸存者偏差（退市股）主分析
├── delist_filter.py      # 面值退市风险过滤
├── sector_analysis.py    # 板块成分分析
├── stock_analyze.py      # 多维个股分析
├── strategy_battle.py    # 经典策略擂台对比
└── figures/              # 结果图表（22 张 PNG）
```

## 数据源

- **baostock** — 全市场日线（含退市股，前复权 `adjustflag="2"`），免费，~0.47 只/秒
- **akshare** — 新浪财务接口 `stock_financial_analysis_indicator`（ROE/毛利率/负债率等）
- **新浪** `stock_zh_a_daily` — 价格数据备用源（东财 `stock_zh_a_hist` 不稳定）

## 环境与依赖

- Python 3.10+（开发环境 3.10.9）
- `akshare` `baostock` `backtrader` `pandas` `numpy` `matplotlib`

```bash
pip install akshare baostock backtrader pandas numpy matplotlib
```

## 运行方式

1. 先跑 `fetch_*.py` 生成数据缓存（`.pkl`，已 gitignore，需本地重新抓取）
2. 再跑对应的分析脚本（如 `value_factor.py`）出 IC 表 + 回测结果 + PNG

> 数据缓存文件（`*.pkl` / `*.csv`）不随仓库分发，需用 `fetch_*.py` 重新抓取。抓取耗时：全市场估值 ~15 分钟，质量因子 ~13 分钟。

## 回测参数约定

- 月度调仓，HOLD=21 天，N_HOLD=50（top50 等权）
- 成本：ROUND_TRIP=0.003（佣金万2.5 + 印花税万5 + 滑点千1）
- 区间：2015–2026
- 估值因子 4 个月年报发布滞后，避免前视偏差

## 免责声明

本项目仅用于量化研究方法学习与验证，不构成任何投资建议。历史回测不等于未来收益，市场有风险，投资需谨慎。
