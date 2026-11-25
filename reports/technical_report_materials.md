# Technical report materials - Agent 3 (dynamic pricing optimization)

Material to reuse in the written report. Every figure and number comes from the files in this repository
(`reports/regression_results.json`, `reports/backtest_*.csv`, `data/processed/pipeline_run.json`). Re-running
the commands in section 8 reproduces them.

**Rule for the whole report:** results that come from a simulation or from an assumption must be labeled as
such. No real profit improvement has been observed.

---

## 1. Problem and role of Agent 3

Recommend, for each product and week, the price that maximizes `profit = (price - cost) x demand`, using:
Agent 1's demand forecast (LSTM), Agent 2's reputation scores (Transformer), costs, current prices and
competitor prices, under realistic pricing constraints.

## 2. Data inventory (what was actually received)

| File | Rows | Content | Issue found |
|---|---|---|---|
| `outputs/demand_forecast.csv` (Agent 1) | 128 | 32 product families x 4 forecast weeks (2017-08-14 to 2017-09-04), `demand_pred` | Keyed by family, not by product |
| `outputs/sales_weekly.csv` (Agent 1) | 7,153 | weekly sales per family, `onpromotion` | **No price column**; 345 zero-sales weeks; 244 flagged outliers |
| `data/processed/reputation_scores.csv` (Agent 2) | 94 | reputation per product **name** | No `product_id`, `category` or attributes (interface not met); 29 products with fewer than 30 reviews |
| `data/processed/pricing_dataset.csv` | 120 | current price, cost, competitor price per product | Single snapshot; margins about 11%-47%; competitor price within about +/-12% of ours; origin to document |

Conclusions: (1) there is no common key between the three sources, (2) the sources describe different
businesses (retail families vs. electronics reviews), (3) the data cannot identify a price elasticity.

## 3. Integration decisions

| Decision | Reason | Where |
|---|---|---|
| Rename `current_price`->`price_current`, `product_cost`->`cost` | Match the project specification | `pricing_config.yaml` |
| Build `product_mapping.csv` (round-robin products to families; seeded assignment of reputation names), label `synthetic` | No real key exists; the file is editable and replaces code changes | `src/pipeline/mapping.py` |
| Split a family's forecast equally between its products | Needed to obtain a per-product demand; changes scale only | `integration.py` |
| Missing reputation (26 products): neutral median value + flag `reputation_available=False` | Keep every product priceable without inventing a score | `integration.py` |
| Invalid cost kept as NaN, price unchanged (configurable) | Never price a product with an unknown cost | `loaders.py`, `constraints.py` |
| `demand_index` = forecast / mean of the last 12 weeks of the family | Lets Agent 1's forecast influence price sensitivity | `integration.py` |

Validation performed on every input: required columns, duplicates, invalid values, negative forecasts (clipped),
scores outside [0,1] (clipped), low-review flags. Originals are never modified.

## 4. Regression

**Model:** `log(demand) = b0 + b1 log(price/usual price) + b2 log(competitor/usual) + b3 reputation + b4 log(baseline demand) + e`.
`b1` is the price elasticity. Prices are relative to each product's usual (training-average) price so cheap and
expensive products are compared fairly. The usual prices come from the training weeks only (no leakage).

**Validation:** chronological. The last 30% of weeks (31 of 104) are held out; expanding-window cross-validation
with 4 folds. Multicollinearity is checked with VIF on the training data; Ridge (RidgeCV, time-ordered folds)
is used only if the maximum VIF exceeds 10.

**Why simulated data:** the real sales data has no prices. A fake price history with a known elasticity (-1.8,
weekly random price moves of about +/-8%, a competitor that partly follows, noise) checks that the method recovers
the truth. Everything it produces carries `is_synthetic = True`.

| Metric (simulated data) | Value |
|---|---|
| Estimated elasticity | -1.769 (true -1.8); 95% interval [-1.800, -1.737] |
| Estimated competitor effect / reputation effect | 0.569 (true 0.6) / 0.502 (true 0.5) |
| Max VIF | 1.93 -> OLS |
| Hold-out R2 (log) | 0.998 (inflated by product size differences) |
| Hold-out R2 vs. baseline | **0.589**; CV folds 0.577 / 0.588 / 0.603 / 0.591 |
| Hold-out MAE / RMSE (log) | 0.081 / 0.102 (baseline 0.127 / 0.159) |
| Hold-out MAE / RMSE (demand units) | 3,427 / 9,558 (baseline 5,448 / 15,326) |

**Real data - promotion effect (not an elasticity):** `log(sales)` on `log(1 + items on promotion)` and seasonality,
per family, last 52 weeks held out. Coefficient +0.109; hold-out R2 -0.057; MAE 0.371 vs 0.418 for the family-average
model. Interpretation: promotions go with higher sales in-sample, but the relationship does not carry over to
the last year (promotion levels grew over time).

## 5. Optimizer

- Demand `D(p) = D0 (p/p0)^e`; profit `(p - cost) D(p)`; analytic optimum without limits `p* = cost e/(1+e)` for `e < -1`.
- Grid search (201 points) inside the allowed interval, plus the current price and `p*` when inside it.
- Output of `optimal_price()`: current and recommended price, demand at both, profit at both, estimated gain, margins,
  assumed elasticity, unconstrained optimum, limiting rule, flags, plain-language reasoning.
- **Elasticity per product (assumption):** `|e| = 3.0 x (1 + gap) x (1 - (rep - 0.714)) x (1 - 0.5 (demand_index - 1))`,
  clipped to [1.2, 6]. Base value from the Lerner rule on the median margin (35%). Tested `-1.5`: all products
  hit the price ceiling.

**Constraint precedence:** hard rules (minimum price, minimum margin, floor, ceiling) are never broken; the
maximum-change rule yields to them (flagged); if the margin and the ceiling collide, the margin wins. A final
**audit** re-checks every row before export.

Result on the project data (480 product-weeks): prices raised in 43% of cases, lowered in 54%, mean change -0.69%;
a pricing rule limited the price in 37% of cases (97 max-change, 64 ceiling, 17 floor). Simulated gain +10.27%
(model estimate).

## 6. Backtest (simulation)

No real outcomes at different prices exist, and evaluating with the optimizer's own model would be circular.
Instead the recommendations are evaluated in worlds where the true elasticity differs from the assumed one.

| True elasticity | Gain | Products gaining | Top-10 share of gain | Others |
|---|---|---|---|---|
| matched (circular reference) | +10.27% | - | - | - |
| -1.5 | +12.25% | 44% | 145% | -8.9% |
| -2.0 | +10.67% | 44% | 130% | -5.1% |
| -3.0 | +9.07% | 82% | 87% | +2.4% |
| -4.0 | +9.52% | 72% | 89% | +1.9% |
| -5.0 | +12.10% | 61% | 118% | -3.7% |
| random product errors (200 runs) | mean +11.45% (5%-95%: +6.0% to +17.6%) | 72% | - | - |

Price statistics: average price 177.71 (static) vs 178.42 (dynamic), -0.69%; mean absolute weekly change 3.8%;
within-product price variation 2.8%. Best categories (mean simulated gain): PLAYERS AND ELECTRONICS 25.6%, PRODUCE 20.2%,
LAWN AND GARDEN 19.4%; weakest: POULTRY 2.4%, DELI 2.1%, AUTOMOTIVE 2.0%.

**Reading:** the strategy gains in total in every scenario, but 87%-145% of the net gain comes from ten large
products; outside them, the strategy loses if the assumed elasticity is wrong (-1.5, -2, -5). Do not state a
profit improvement as a result.

### Figures (`reports/figures/`)

| File | Caption to use |
|---|---|
| `backtest_1_gain_vs_true_elasticity.png` | Simulated profit gain if the true elasticity differs from the assumed -3.0, for all products and for all products except the ten largest gainers. |
| `backtest_2_monte_carlo.png` | Distribution of the simulated total gain over 200 runs with random product-level elasticity errors. |
| `backtest_3_by_category.png` | Average simulated gain per product family. |
| `backtest_4_price_changes.png` | Distribution of recommended price changes (model output, not an outcome). |

## 7. Assumptions register

| Assumption | Value | Where to change | Effect if wrong |
|---|---|---|---|
| Base elasticity | -3.0 | `optimizer.base_elasticity` | Drives every recommendation (see backtest) |
| Gap / reputation / demand sensitivities | 1.0 / 1.0 / 0.5 | `optimizer.*_sensitivity` | Change which products rise or fall |
| Constant-elasticity demand | - | `optimizer.py` | Misses saturation, thresholds, competitor reactions |
| Equal split of family forecast | - | `integration.py` | Scale of profit only |
| Synthetic product mapping | seed 42 | `product_mapping.csv` | Product-level combinations are illustrative |
| Pricing rules | margin 10%, +/-20% band, +/-10% per update | `constraints` | Bound the price moves |

## 8. Reproducibility

```
python -m pip install -r requirements-pricing.txt
python -m src.pipeline.pipeline          # integration + regression + optimization + export
python -m src.pricing.regression         # regression report
python -m src.pricing.optimizer          # optimizer summary and elasticity sensitivity
python -m src.pricing.backtest           # backtest tables and figures
python -m pytest tests/test_data_loaders.py tests/test_integration.py tests/test_regression.py tests/test_constraints.py tests/test_optimizer.py tests/test_backtest.py tests/test_pipeline.py tests/test_api.py tests/test_dashboard.py -q
```
Random seeds (simulation, mapping, Monte Carlo) are fixed in `pricing_config.yaml` (seed 42).

## 9. Limitations to state in the report

Assumed elasticity; synthetic links between datasets; constant-elasticity form; no stock or competitor reaction;
competitor price used only to adjust sensitivity (not as an anchor); equal split of family forecasts; volume-weighted
results hide small-product losses; regression validated only on simulated data; forecast errors propagate;
only four forecast weeks, so no real out-of-sample profit test.

## 10. Suggested next steps (not implemented)

Collect real price-and-sales history (or run a price experiment) to estimate elasticity with the existing
regression code; replace the synthetic mapping with real keys; ask Agent 2 to output `product_id`, `category` and
attributes; add Lasso and competitor-aware price anchoring; evaluate on a live A/B test.
