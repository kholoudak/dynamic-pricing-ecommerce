# Demonstration script (about 4 minutes)

**Before you start (once, 2 minutes before the session)**

```
python -m src.pipeline.pipeline
python -m streamlit run src/pipeline/dashboard.py --server.headless true
```
Open http://localhost:8501. Keep the page on the **Overview** tab, category filter empty, week `2017-08-14`.
Optional second terminal for the API: `python -m src.pipeline.api` (docs at http://127.0.0.1:8000/docs).

All numbers below are the ones the dashboard shows with the project data. If you re-run the pipeline with
different inputs, read the numbers off the screen instead.

---

## Timeline

| Time | What you do | What you say |
|---|---|---|
| 0:00 - 0:30 | Show the page title and the yellow banner. | "The system recommends a price for each product every week. It combines Agent 1's LSTM demand forecast, Agent 2's reputation scores, product costs and competitor prices. **One important point first: our data has no historical prices, so the price sensitivity is an assumption. Every profit figure you will see is a simulation, not a measured result.**" |
| 0:30 - 1:15 | Point to the KPI cards. | "120 products analyzed. Prices go from an average of 177.71 to 177.52, so the average barely moves. Predicted demand is about 5.4 million units. 50 prices are raised, 65 lowered, 5 unchanged. In 75 cases a pricing rule limited the move. The simulated profit goes from 283.5 M to 306.9 M model units, +8.3%, **under our assumed elasticity**." |
| 1:15 - 2:30 | Open **Product detail**. Select `24 - MEATS`. | See "Product A" below. |
| 2:30 - 3:15 | Select `76 - GROCERY I`. | See "Product B" below. |
| 3:15 - 3:45 | Open **Model and evidence**. | "This tab is where we are honest about the limits. The regression recovers a known elasticity on simulated data, so the method works. On the real data there are no prices, so we assume the elasticity and test what happens if we are wrong. The backtest shows the gain depends on a few large products." Point at the yellow bars. |
| 3:45 - 4:15 | Click **Refresh data and recalculate prices** in the sidebar (about 2-4 s). Optionally switch to the API docs page. | "The refresh button re-runs the whole pipeline: validation, merge, regression, optimization, constraints, and an audit that refuses to export a price that breaks a rule. The same pipeline is available through the API: `GET /price/{id}` and `POST /refresh`." |

---

## Product A - `24 - MEATS`: the price rises, no rule needed

| | Before | After |
|---|---|---|
| Price | 369.47 | **401.79** (+8.7%) |
| Predicted demand | 33.9 K | 25.2 K (-25.8%) |
| Expected weekly profit | 2.7 M | 2.9 M (+3.9%) |
| Margin on price | 21.9% | 28.2% |

Say: "The unit cost is 288.65, so the current margin is only 21.9%. The model estimates this product's price
sensitivity at -3.55: it is priced 2.2% above the competitor (361.34) and its reputation is below typical
(0.59), which makes customers more sensitive. Even so, with such a thin margin, a higher price earns more
profit despite losing about a quarter of the volume. The recommended price **is** the mathematical
optimum (401.79), so no rule limited it."

Show the weekly chart: 401.79, 399.99, 405.16, 401.55 (small adjustments as the forecast changes).

## Product B - `76 - GROCERY I`: the price falls, limited by a rule

| | Before | After |
|---|---|---|
| Price | 74.53 | **67.08** (-10.0%) |
| Predicted demand | 389.7 K | 559.2 K (+43.5%) |
| Expected weekly profit | 12.6 M | 14.0 M (+10.5%) |
| Margin on price | 43.5% | 37.2% |

Say: "Here the margin is high (43.5%, unit cost 42.11) and we are 1.3% above the competitor (73.56). The model
says lowering the price brings so much extra volume that profit rises. Without any limit it would go down to
59.45 (-20%), but the rule 'maximum change per update = 10%' stops it at 67.08. In the following weeks it
continues to 60.38, then stays near 60." Show the reasoning box: it says this in plain language.

---

## Likely questions and honest answers

**Why is the elasticity assumed and not estimated?**
There are no historical prices in the data, so it cannot be measured. We validated the regression on
simulated data with a known answer (estimated -1.77 vs true -1.8) and we test how results change if the
assumption is wrong.

**Where does -3.0 come from?**
From the Lerner rule (elasticity = -1 / margin) applied to the median margin of 35%. We first tried -1.5; it
pushed every product to the price ceiling, which was not a meaningful result.

**Is the +8% / +10% profit real?**
No. It is a model estimate under our assumptions. In the backtest the gain stays positive in all tested
scenarios but comes mostly from about ten large products; for many others it can be negative if the
assumed elasticity is wrong.

**How do Agent 1 and Agent 2 influence the price?**
Agent 1's forecast sets the demand level and, through the "demand index" (forecast versus the last 12 weeks),
lowers price sensitivity when demand is strong. Agent 2's reputation lowers sensitivity for well-rated
products. Both effects are assumptions with weights in the configuration file.

**Why does the price of a product that is cheaper than its competitor sometimes go down?**
The competitor price only changes the assumed sensitivity; it is not a cap or a target. This is a known
limitation (see the README).

**Are the links between products, families and reputation real?**
No. The three datasets describe different businesses, so the mapping is synthetic and saved in
`product_mapping.csv`. It can be replaced by a real one without changing the code.

**What if a cost is missing?**
The price is kept unchanged and flagged (configurable: it can use a default cost ratio instead).

**Why was Ridge not used?**
The multicollinearity check (VIF) was 1.93, below the threshold of 10. Ridge switches on automatically
above it, and a test proves that.

**What would you do with real data?**
Replace the mapping with real keys, collect price changes and sales to estimate the elasticity with the same
regression code, and compare recommended and static prices in a real experiment.

---

## If something goes wrong during the demo

| Problem | Fix |
|---|---|
| Dashboard says "No recommendations yet" | Click the Refresh button, or run `python -m src.pipeline.pipeline` |
| Page does not open | Check the terminal shows `Local URL: http://localhost:8501`; open it manually |
| Port already in use | Stop the old terminal with Ctrl+C, or add `--server.port 8502` and open that port |
| Numbers differ from this script | Normal if the input files changed; read the numbers off the screen |
