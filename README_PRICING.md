# Dynamic Pricing Engine (Agent 3)

Agent 3 of the project **Dynamic Pricing & Competition in E-Commerce**. It takes the demand forecast
of Agent 1 (LSTM), the reputation scores of Agent 2 (Transformer) and the product prices and costs,
and recommends a price for every product and week. It comes with a pipeline, an API and a dashboard.

> **Read this first - what the results mean**
> The available data contains **no historical prices**, so the price sensitivity (elasticity) of the
> products **cannot be measured**. The optimizer uses an **assumed** elasticity (`-3.0`, adjusted per
> product). Every profit and gain figure in this project is therefore a **simulated estimate under
> stated assumptions**, not an observed improvement. The links between the 120 priced products, the
> 32 forecast families and the reputation scores are **synthetic** (the three datasets describe
> different businesses).

---

## 1. What is implemented

| Task | Status | Where |
|---|---|---|
| Data integration and validation | Implemented | `src/pipeline/loaders.py`, `mapping.py`, `integration.py` |
| Log-log price-elasticity regression (OLS, Ridge when VIF > 10), chronological validation | Implemented, run on **simulated** data (real data has no prices) | `src/pricing/regression.py`, `synthetic.py` |
| Promotion-effect regression on the real sales history | Implemented (descriptive only, not an elasticity) | `src/pricing/regression.py` |
| Price optimizer (`optimal_price`) with explanation | Implemented | `src/pricing/optimizer.py` |
| Configurable pricing constraints | Implemented | `src/pricing/constraints.py`, `pricing_config.yaml` |
| Backtest vs. static price | Implemented as a **simulation** (no real outcomes exist) | `src/pricing/backtest.py` |
| Pipeline with constraint audit | Implemented | `src/pipeline/pipeline.py` |
| FastAPI (`GET /price/{product_id}`, `POST /refresh`) | Implemented | `src/pipeline/api.py` |
| Streamlit dashboard | Implemented | `src/pipeline/dashboard.py` |
| Tests | 72 tests (see section 10) | `tests/` |

**Not implemented**
- Estimating a real price elasticity (impossible without historical prices).
- Lasso (only Ridge is implemented as the multicollinearity remedy).
- Product attributes from Agent 2 (its file has none), competitor reactions, stock levels, long-horizon planning.

---

## 2. Files added by Agent 3

```
pricing_config.yaml          all paths, constraints and assumptions (nothing hard-coded in the code)
requirements-pricing.txt     what Agent 3 needs to run (works on Python 3.13, no TensorFlow/PyTorch)
src/pricing/
    config.py                loads pricing_config.yaml
    regression.py            elasticity model + promotion model + report
    synthetic.py             SIMULATED price experiment with a known elasticity
    constraints.py           pricing rules (floor, ceiling, margin, max change, missing cost)
    optimizer.py             optimal_price() and recommend_all()
    backtest.py              simulated comparison with the static strategy + charts
src/pipeline/
    loaders.py               read and validate the four input files
    mapping.py               product <-> family <-> reputation mapping (synthetic, saved as a file)
    integration.py           merge everything into the master table
    pipeline.py              end-to-end run + audit + export
    api.py                   FastAPI service
    dashboard.py             Streamlit dashboard
tests/                       tests for each module (+ conftest.py with mock data)
data/processed/              recommended_prices.csv, pipeline_run.json, product_mapping.csv, master_table.csv
reports/                     figures, backtest tables, regression_results.json, demo_script.md, technical_report_materials.md
```

The code of Agents 1 and 2 is **not modified**. Their output files are only read.

---

## 3. Installation

Python 3.10 or newer. From the project folder:

```
python -m pip install -r requirements-pricing.txt
```

(`requirements.txt` also contains the packages of Agents 1 and 2. Some of its pinned versions, such as
`numpy==1.26.4` and `torch==2.3.1`, do not install on Python 3.13. Agent 3 does not need them.)

---

## 4. How to run

Always run commands **from the project root** (the folder that contains `pricing_config.yaml`).

| Goal | Command |
|---|---|
| Run the whole pipeline and export `recommended_prices.csv` | `python -m src.pipeline.pipeline` |
| Start the API (docs at http://127.0.0.1:8000/docs) | `python -m src.pipeline.api` |
| Start the dashboard (open http://localhost:8501) | `python -m streamlit run src/pipeline/dashboard.py --server.headless true` |
| Regression report | `python -m src.pricing.regression` |
| Optimizer summary and sensitivity table | `python -m src.pricing.optimizer` |
| Backtest and charts | `python -m src.pricing.backtest` |
| Check the inputs | `python -m src.pipeline.loaders` |
| All Agent 3 tests | `python -m pytest tests/test_data_loaders.py tests/test_integration.py tests/test_regression.py tests/test_constraints.py tests/test_optimizer.py tests/test_backtest.py tests/test_pipeline.py tests/test_api.py tests/test_dashboard.py -q` |

Run the pipeline once before starting the API or the dashboard (or use their refresh button / endpoint).

**Expected pipeline output:** four `[ok]` lines, then `exported 480 rows to recommended_prices.csv` and
`120 products x 4 weeks = 480 recommendations`.

---

## 5. Data and interface

| File | Producer | Used columns |
|---|---|---|
| `outputs/demand_forecast.csv` | Agent 1 | `product_id` (a product **family**), `week`, `demand_pred` |
| `outputs/sales_weekly.csv` | Agent 1 | `product_id`, `week`, `sales`, `onpromotion`, `is_outlier` |
| `data/processed/reputation_scores.csv` | Agent 2 | `product_name`, `reputation_score`, `n_reviews` |
| `data/processed/pricing_dataset.csv` | prices and costs (origin to be documented by the team) | `product_id`, `current_price`, `product_cost`, `competitor_price` |

Output: `data/processed/recommended_prices.csv`. The first five columns are the interface agreed in the
project plan: `product_id, week, price_current, price_recommended, margin_gain`. Further columns hold
demand, profit, margins, reputation, the assumed elasticity, the limiting rule, flags and a plain-language
`reasoning`.

### Compatibility issues found (and how they are handled)

1. **No common product key.** Agent 1 forecasts 32 families, the price file has 120 numbered products, and
   Agent 2 scores product names. Agent 3 builds `data/processed/product_mapping.csv` (round-robin
   products to families, seeded assignment of reputation names) and labels it `synthetic`. **If real keys
   appear, edit that file; no code changes are needed.**
2. **Agent 2 does not output `product_id`, `category` or attributes** (the agreed interface). 26 of the 120
   products therefore have no reputation score; they get a neutral value and are flagged
   `reputation_available = False`. 29 of Agent 2's 94 scores rest on fewer than 30 reviews (flagged).
3. **The datasets describe different businesses** (retail families vs. electronics reviews), so
   product-level combinations are not meaningful; the pipeline mechanics are.
4. **No price history** in the sales data, hence the assumed elasticity.
5. Column names differ from the specification (`current_price` / `product_cost`); renamed in the config.

---

## 6. The pricing model

**Demand:** `D(p) = D0 * (p / p0)^e`, where `p0` is the current price, `D0` is Agent 1's forecast at the
current price (split equally between the products of a family) and `e` is the elasticity (negative).

**Profit:** `profit(p) = (p - cost) * D(p)` (per week, in model units).

**Elasticity per product (all assumptions, set in `pricing_config.yaml`):**

```
|e| = |-3.0| x (1 + gap_sens x gap) x (1 - rep_sens x (reputation - 0.714)) x (1 - dem_sens x (demand_index - 1))
```
- `gap` = current price / competitor price - 1: pricier than the competitor means more price-sensitive.
- `reputation` (Agent 2): better reputation means less price-sensitive. Missing reputation is neutral.
- `demand_index` = forecast / average of the last 12 weeks (Agent 1): stronger demand means less price-sensitive.
- The result is kept between -1.2 and -6. The base value `-3.0` is calibrated with the Lerner rule
  (`e = -1 / margin`) on the median margin (35%). Testing `-1.5` pushed all products to the price ceiling.

**Search:** a 201-point grid between the lowest and highest allowed price, plus the current price and the
analytic optimum `cost x e / (1 + e)`. The best profit wins.

**Constraints** (`pricing_config.yaml`, section `constraints`):

| Rule | Default | Type |
|---|---|---|
| Minimum price | 0.01 (never zero or negative) | hard |
| Minimum margin | price >= cost x 1.10 | hard |
| Price floor / ceiling | 80% / 120% of the current price | hard |
| Maximum change per update | +/-10% versus the previous week's price | soft: yields to the hard rules |
| Missing or invalid cost | keep the current price (or use a default cost ratio) | policy |

If two hard rules collide (margin above the ceiling), the minimum margin wins. The pipeline re-checks every
recommendation after optimizing (**audit**) and stops without exporting if any rule is broken.

---

## 7. Model evaluation

All numbers come from `reports/regression_results.json` and the `reports/backtest_*.csv` files.

### Elasticity regression - SIMULATED data with a known answer
`log(demand) = b0 + b1 log(price / usual price) + b2 log(competitor / usual) + b3 reputation + b4 log(baseline demand)`.
Prices are measured relative to each product's usual price (from the training weeks only). The last 30% of
the weeks (31 of 104) are held out; nothing is shuffled.

| Result | Value |
|---|---|
| True elasticity (built into the simulation) | -1.8 |
| Estimated elasticity (OLS) | -1.769, 95% interval [-1.800, -1.737] |
| Multicollinearity (max VIF) | 1.93, below 10, so plain OLS (Ridge is triggered above 10; tested) |
| Hold-out R2 on the log scale | 0.998 (inflated: products differ hugely in size; the baseline alone gets 0.996) |
| **Hold-out R2 versus the baseline (the honest measure)** | **0.589** (temporal CV folds: 0.577, 0.588, 0.603, 0.591) |
| Hold-out MAE / RMSE (log scale) | 0.081 / 0.102 (baseline "ignore price": 0.127 / 0.159) |

This shows that **the method works**. It says nothing about the elasticity of real customers.

### Promotion effect - REAL sales history (not a price elasticity)
Sales vs. the number of items on promotion, per family, last 52 weeks held out:
coefficient +0.109 (in-sample significant), **hold-out R2 = -0.057**, MAE 0.371 versus 0.418 for predicting
each family's average. The relationship exists in-sample but does not forecast well out of sample
(promotions grew over time, so the level shifts).

### Backtest - SIMULATED (no real outcomes at different prices exist)
Dynamic recommendations versus keeping current prices, simulated with several "true" elasticities
(the optimizer assumed -3.0):

| True elasticity | Profit gain | Products gaining | Share of the gain from the top 10 products | Gain of all other products |
|---|---|---|---|---|
| -1.5 | +12.25% | 44% | 145% | -8.9% |
| -2.0 | +10.67% | 44% | 130% | -5.1% |
| -3.0 | +9.07% | 82% | 87% | +2.4% |
| -4.0 | +9.52% | 72% | 89% | +1.9% |
| -5.0 | +12.10% | 61% | 118% | -3.7% |

With random product-level errors (200 runs) the mean gain is +11.45% (5%-95%: +6.0% to +17.6%).
**But the gain is concentrated in a few large products**; for many others the strategy loses money if the
assumed elasticity is wrong. Prices stay close to current ones on average (-0.69%), with a mean absolute
weekly change of 3.8%.

**Correct wording:** "Under the stated assumptions, the strategy raises total simulated profit in all tested
scenarios, mainly through a few large product families. This is not evidence of a real profit improvement."

---

## 8. Limitations

- **Assumed elasticity.** The central parameter is not estimated from data. Results change with it
  (`python -m src.pricing.optimizer` prints a sensitivity table).
- **Constant-elasticity demand** in logs; no stock, capacity, seasonality of elasticity or competitor reaction.
- **Synthetic links** between products, families and reputation scores. Product-level results are illustrative.
- **Competitor price is not an anchor.** It only changes the assumed price sensitivity. It is not used as a cap or a
  target, so a product that is already cheaper than its competitor can still be lowered.
- **Equal split** of a family's forecast between its products (changes the scale of profit, not the optimal price).
- **Volume weighting** hides small-product losses (see the backtest table).
- **Regression on simulated data:** random price changes are easier to learn than real, purposeful pricing
  (which would bias real estimates). Competitor and reputation effects are held fixed in the optimizer.
- **Forecast quality** is Agent 1's responsibility (see its evaluation files); errors flow into profit estimates.
- **Short horizon:** only 4 forecast weeks exist, so there is no real out-of-sample profit comparison.

---

## 9. API quick reference

| Request | Result |
|---|---|
| `GET /price/{product_id}` (optional `?week=YYYY-MM-DD`) | current and recommended price, demand, profit, estimated gain, reputation, reasoning, weekly schedule |
| `POST /refresh` (optional JSON `{"rebuild_mapping": false, "run_regression": true}`) | re-runs the pipeline and returns a summary |

Errors: 404 unknown product or week, 422 invalid input, 503 nothing exported yet, 409 refresh already running,
500 pipeline failure (the previous results keep being served). Interactive documentation: `/docs`.

---

## 10. Tests

72 tests, all runnable without Agents 1 and 2: `test_data_loaders` (8), `test_integration` (5), `test_regression` (8),
`test_constraints` (8), `test_optimizer` (12), `test_backtest` (6), `test_pipeline` (6), `test_api` (13),
`test_dashboard` (6). The pipeline test runs end to end on small mock files, which shows how real files replace
mock files without changing the code. The dashboard tests run the real app headless.

---

## 11. Changing the configuration

Edit `pricing_config.yaml`, then run the pipeline again. Typical changes: `constraints` (margin, floor, ceiling,
maximum change), `optimizer.base_elasticity` and the three sensitivities, `paths` (to point at other input files).

---

More documents: `reports/demo_script.md` (3-5 minute demonstration) and
`reports/technical_report_materials.md` (material for the written report).
