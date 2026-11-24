"""FastAPI service for the pricing engine (Agent 3, Phase 7).

The API contains NO pricing logic. It only
  * reads the recommendations exported by the pipeline (recommended_prices.csv), and
  * calls the pipeline again when asked to refresh.

Run:   python -m src.pipeline.api            (or:  uvicorn src.pipeline.api:app --reload)
Docs:  http://127.0.0.1:8000/docs            (interactive page where you can try both endpoints)
"""
import math
import threading
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.pipeline.loaders import DataValidationError
from src.pipeline.pipeline import (
    SIMULATION_NOTE, PipelineError, load_recommendations, load_run_info, run_pipeline,
)
from src.pricing.config import load_config

DESCRIPTION = """
Dynamic pricing recommendations for an e-commerce catalogue.

**Important:** the recommendations come from a model whose price elasticity is an *assumption*
(the available data contains no historical prices). Profit and gain values are **simulated
estimates**, not observed results.
"""


# ----------------------------- response models (what the API returns) -----------------------------

class WeekPrice(BaseModel):
    week: date = Field(description="Forecast week (Monday)")
    price_recommended: float
    price_change_pct: float = Field(description="Change versus the CURRENT price, in %")


class PriceResponse(BaseModel):
    product_id: str
    category: str | None = Field(None, description="Product family used to link the product to Agent 1's forecast")
    week: date = Field(description="Week this recommendation applies to")
    price_current: float
    price_recommended: float
    price_change_pct: float
    demand_current: float | None = Field(None, description="Predicted demand at the current price (Agent 1)")
    demand_recommended: float | None = Field(None, description="Predicted demand at the recommended price")
    profit_current: float | None = Field(None, description="Expected weekly profit at the current price (model units)")
    profit_recommended: float | None = Field(None, description="Expected weekly profit at the recommended price")
    estimated_margin_gain: float | None = Field(None, description="profit_recommended - profit_current (simulated)")
    margin_pct_current: float | None = None
    margin_pct_recommended: float | None = None
    competitor_price: float | None = None
    reputation_score: float | None = Field(None, description="Agent 2 reputation score in [0, 1]")
    reputation_available: bool = Field(description="False when no reputation could be linked (neutral value used)")
    elasticity_used: float | None = Field(None, description="Assumed price elasticity for this product")
    binding_constraint: str | None = Field(None, description="Pricing rule that limited the recommendation, if any")
    status: str = Field(description="'ok' or 'no_recommendation' (e.g. missing cost: current price kept)")
    flags: list[str] = []
    reasoning: str = Field(description="Plain-language explanation of the recommendation")
    schedule: list[WeekPrice] = Field(description="Recommended price for every forecast week")
    simulated: bool = True
    note: str = SIMULATION_NOTE


class RefreshRequest(BaseModel):
    rebuild_mapping: bool = Field(False, description="Rebuild product_mapping.csv instead of re-using it")
    run_regression: bool = Field(True, description="Also re-run the elasticity validation (about 3 s)")


class RefreshResponse(BaseModel):
    status: str
    generated_at: str
    n_products: int
    n_weeks: int
    n_rows: int
    simulated_gain_pct: float
    mean_price_change_pct: float
    duration_seconds: float
    warnings: list[str]
    note: str = SIMULATION_NOTE


# ----------------------------- helpers -----------------------------

def _clean(value):
    """JSON cannot carry NaN: turn NaN/NaT into None and numpy types into Python types."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return None
    return value.item() if hasattr(value, "item") else value


def _row_to_response(row, schedule_df):
    schedule = [WeekPrice(week=r.week.date(), price_recommended=float(r.price_recommended),
                          price_change_pct=float(r.price_change_pct)) for r in schedule_df.itertuples()]
    flags = [f for f in str(row["flags"]).split(";") if f]
    return PriceResponse(
        product_id=str(row["product_id"]), category=_clean(row["category"]), week=row["week"].date(),
        price_current=float(row["price_current"]), price_recommended=float(row["price_recommended"]),
        price_change_pct=float(row["price_change_pct"]),
        demand_current=_clean(row["demand_current"]), demand_recommended=_clean(row["demand_recommended"]),
        profit_current=_clean(row["profit_current"]), profit_recommended=_clean(row["profit_recommended"]),
        estimated_margin_gain=_clean(row["margin_gain"]),
        margin_pct_current=_clean(row["margin_pct_current"]), margin_pct_recommended=_clean(row["margin_pct_recommended"]),
        competitor_price=_clean(row["competitor_price"]), reputation_score=_clean(row["reputation_score"]),
        reputation_available=bool(row["reputation_available"]), elasticity_used=_clean(row["elasticity_used"]),
        binding_constraint=_clean(row["binding_constraint"]), status=str(row["status"]), flags=flags,
        reasoning=str(row["reasoning"]), schedule=schedule)


# ----------------------------- the application -----------------------------

def create_app(cfg=None):
    """Build the FastAPI app. ``cfg`` lets tests point the API at mock files."""
    cfg = cfg or load_config()
    app = FastAPI(title="Dynamic Pricing API", version="1.0.0", description=DESCRIPTION)
    app.state.df = None
    app.state.lock = threading.Lock()          # only one refresh at a time

    def recommendations():
        if app.state.df is None:
            try:
                app.state.df = load_recommendations(cfg)
            except FileNotFoundError:
                raise HTTPException(503, "No recommendations available yet. Call POST /refresh first.")
        return app.state.df

    @app.get("/", include_in_schema=False)
    def root():
        """Send people who open the bare address to the interactive documentation."""
        return RedirectResponse("/docs")

    @app.get("/price/{product_id}", response_model=PriceResponse, tags=["pricing"],
             summary="Recommended price for one product",
             responses={404: {"description": "Unknown product or week outside the forecast horizon"},
                        422: {"description": "Invalid product_id or week format"},
                        503: {"description": "No recommendations yet - call POST /refresh"}})
    def get_price(
        product_id: str = Path(..., min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_\-]+$",
                               description="Product identifier, e.g. 1", examples=["1"]),
        week: date | None = Query(None, description="Forecast week (YYYY-MM-DD). Default: the first forecast week."),
    ):
        """Return current and recommended price, predicted demand, expected profit, the estimated
        gain and the reasoning for one product, plus the recommended price of every forecast week."""
        df = recommendations()
        product = df[df["product_id"] == product_id].sort_values("week")
        if product.empty:
            raise HTTPException(404, f"Product '{product_id}' not found.")
        if week is None:
            row = product.iloc[0]
        else:
            match = product[product["week"].dt.date == week]
            if match.empty:
                available = ", ".join(str(d.date()) for d in product["week"])
                raise HTTPException(404, f"No recommendation for week {week}. Available weeks: {available}.")
            row = match.iloc[0]
        return _row_to_response(row, product)

    @app.post("/refresh", response_model=RefreshResponse, tags=["pipeline"],
              summary="Reload the datasets and recalculate all recommended prices",
              responses={409: {"description": "A refresh is already running"},
                         500: {"description": "The pipeline failed; the previous recommendations stay available"}})
    def refresh(body: RefreshRequest | None = None):
        """Re-run the whole pipeline (load, validate, merge, regression, optimize, constraints, audit, export).
        If it fails, the previously exported recommendations keep being served."""
        body = body or RefreshRequest()
        if not app.state.lock.acquire(blocking=False):
            raise HTTPException(409, "A refresh is already running. Try again in a few seconds.")
        try:
            result = run_pipeline(cfg, rebuild_mapping=body.rebuild_mapping, run_regression=body.run_regression)
            app.state.df = load_recommendations(cfg)
        except (PipelineError, DataValidationError, FileNotFoundError) as exc:
            raise HTTPException(500, f"Refresh failed: {exc}")
        finally:
            app.state.lock.release()
        m = result.metadata
        return RefreshResponse(status="ok", generated_at=m["generated_at"], n_products=m["n_products"],
                               n_weeks=m["n_weeks"], n_rows=m["n_rows"],
                               simulated_gain_pct=m["simulated_gain_pct"],
                               mean_price_change_pct=m["mean_price_change_pct"],
                               duration_seconds=m["duration_seconds"], warnings=result.warnings)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
