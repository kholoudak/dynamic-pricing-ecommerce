# Dynamic Pricing & Competition in E-Commerce

Projet universitaire n°8 — Intelligent pricing and demand forecasting using machine learning.

This project is composed of multiple agents working together to analyze product demand, reputation, and pricing decisions in an e-commerce environment.

## Agents Overview

| Agent | Role | Technique |
|---|---|---|
| Agent 1 | Weekly demand forecasting | LSTM |
| Agent 2 | Reputation analysis from reviews | Transformer |
| Agent 3 | Dynamic pricing and price optimization | Linear Regression |

---

# Agent 1 — Demand Forecasting (LSTM)

Agent 1 predicts weekly demand for each product category, 1 to 4 weeks ahead, using an LSTM neural network (TensorFlow/Keras).

Its predictions are transmitted to Agent 3 through the integration file:

`outputs/demand_forecast.csv`

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn

## Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Results

Error percentage (WAPE, lower is better), 32 product categories, 4 forecast horizons combined.

| Model | Validation | Test |
|---|---:|---:|
| Ensemble of 5 LSTMs | 8.77% | 7.01% |
| Hybrid (0.5 LSTM + 0.5 seasonal naive) | 7.32% | 8.63% |
| Naive (last week) | 11.42% | 7.95% |
| Moving average (4 weeks) | 10.00% | **6.20%** |
| Seasonal naive (same week last year) | 8.48% | 12.66% |

The LSTM outperforms the naive model on both periods. However, the moving average performs better on the test set. The differences are relatively small and should be interpreted cautiously because the evaluation uses a single 26-week test period.

Detailed results and limitations are available in `RAPPORT_AGENT1.md`.

## Project Structure

```text
dynamic-pricing-ecommerce/
│
├── config.yaml
├── requirements.txt
├── README.md
├── README_PRICING.md
├── CONTRACT.md
├── RAPPORT_AGENT1.md
│
├── data/
│   ├── raw/
│   │   └── Kaggle datasets
│   └── processed/
│       └── processed data and sequences
│
├── notebooks/
│   ├── 01_exploration
│   ├── 02_cleaning
│   └── 03_training_and_results
│
├── models/
│   ├── LSTM models
│   ├── model metadata
│   ├── ensemble metadata
│   └── feature scaler
│
├── outputs/
│   ├── demand_forecast.csv
│   ├── evaluations
│   └── graphs
│
├── src/
│   └── forecast/
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── feature_engineering.py
│       ├── sequence_generator.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── experiments.py
│       ├── ensemble.py
│       ├── predict.py
│       ├── export_forecast.py
│       └── final_evaluation.py
│
└── tests/
    └── unit tests
```

## Installation — Windows 11

Using PowerShell:

```powershell
git clone https://github.com/kholoudak/dynamic-pricing-ecommerce.git

cd dynamic-pricing-ecommerce

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install --default-timeout=120 --retries 10 -r requirements.txt
```

Version tested: Python 3.13.5, TensorFlow 2.21.0.

The requirements file specifies minimum versions. To save your exact environment:

```bash
pip freeze > requirements-lock.txt
```

## Dataset

Agent 1 uses the Kaggle dataset:

**Store Sales — Time Series Forecasting (Favorita)**

The following files should be placed inside `data/raw/`:

- `train.csv`
- `holidays_events.csv`
- `stores.csv`
- `oil.csv`
- `transactions.csv`

The model uses `train.csv` and `holidays_events.csv`. The other files are used for exploration.

Important dataset characteristics:

- `product_id` corresponds to the product family/category, not an individual SKU.
- Sales are aggregated across the 54 stores.
- The dataset does not contain a product price column.

## Agent 1 — Pipeline Execution

Execute the pipeline in the following order:

```powershell
python -m src.forecast.preprocessing

python -m src.forecast.sequence_generator

python -m src.forecast.train

python -m src.forecast.evaluate --split val

python -m src.forecast.ensemble

python -m src.forecast.export_forecast

python -m src.forecast.final_evaluation --final
```

### Pipeline steps

| Step | Module | Description |
|---|---|---|
| 1 | `preprocessing` | Cleaning and weekly aggregation |
| 2 | `sequence_generator` | Feature engineering, scaling and sequence generation |
| 3 | `train` | Training the main LSTM |
| 4 | `evaluate` | Evaluation and comparison with baseline models |
| 5 | `ensemble` | Training and combining 5 LSTM models |
| 6 | `export_forecast` | Exporting demand predictions |
| 7 | `final_evaluation` | Final test evaluation |

Optional experiments:

```powershell
python -m src.forecast.experiments
```

**Important:** Do not run `sequence_generator` on new data without retraining the models, because it rewrites `models/feature_scaler.json` and recalculates the data split dates.

The test dataset should only be used for final evaluation.

## Using the Demand Forecast

From Python:

```python
from src.forecast.predict import predict_demand

# Forecast demand for the next week
predict_demand("GROCERY I", 1)

# Forecast demand for a specific date
predict_demand("GROCERY I", "2017-08-21")
```

The forecast horizon can be:

- Week 1 to 4 after the last observed week.
- A date corresponding to the desired forecast week.

An explicit `ValueError` is raised for an unknown product or invalid forecast horizon.

From the command line:

```powershell
python -m src.forecast.predict
```

Forecast a specific product:

```powershell
python -m src.forecast.predict --product "GROCERY I" --week 1
```

## Integration with Agent 3

Agent 1 exports the predictions to:

`outputs/demand_forecast.csv`

The file contains one row per product and forecast week, for a total of 32 × 4 = 128 rows.

| Column | Type | Description |
|---|---|---|
| `product_id` | String | Product family/category |
| `week` | Date | Monday of the forecast week |
| `demand_pred` | Number ≥ 0 | Predicted weekly sales across all stores |
| `horizon` | Integer (1