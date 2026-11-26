"""Streamlit dashboard for the pricing engine (Agent 3, Phase 8).

Run (from the project root):   streamlit run src/pipeline/dashboard.py

The dashboard reads data/processed/recommended_prices.csv (created by the pipeline) and can
re-run the pipeline with the "Refresh" button. It does not need the API to be running.
All numbers shown for profit and gain are SIMULATED estimates (see the banner and the last tab).
"""
import sys
from pathlib import Path

# `streamlit run` does not add the project root to Python's path, so we do it here.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.pipeline.pipeline import load_recommendations, load_run_info, run_pipeline
from src.pricing.config import load_config, resolve_path

BANNER = ("SIMULATED RESULTS - the available data has no historical prices, so the price sensitivity "
          "(elasticity) is an ASSUMPTION. Profit and gain figures are model estimates, not observed results.")


# --------------------------------------------------------------------------------------
# Pure data functions (no Streamlit inside, so they are easy to test)
# --------------------------------------------------------------------------------------

def filter_data(df, categories=None, week=None):
    """Keep only rows with a recommendation, for the chosen categories and week."""
    out = df[df["status"] == "ok"]
    if categories:
        out = out[out["category"].isin(categories)]
    if week is not None:
        out = out[out["week"] == pd.Timestamp(week)]
    return out


def compute_kpis(f):
    """Headline numbers for the filtered table."""
    if f.empty:
        return None
    rep = f[f["reputation_available"]]
    profit_cur, profit_rec = f["profit_current"].sum(), f["profit_recommended"].sum()
    return {
        "n_products": int(f["product_id"].nunique()),
        "avg_price_current": float(f["price_current"].mean()),
        "avg_price_recommended": float(f["price_recommended"].mean()),
        "avg_price_change_pct": float((f["price_recommended"] / f["price_current"] - 1).mean() * 100),
        "demand_current": float(f["demand_current"].sum()),
        "demand_recommended": float(f["demand_recommended"].sum()),
        "profit_current": float(profit_cur),
        "profit_recommended": float(profit_rec),
        "gain": float(f["margin_gain"].sum()),
        "gain_pct": float(f["margin_gain"].sum() / profit_cur * 100) if profit_cur else 0.0,
        "n_raised": int((f["price_change_pct"] > 0.5).sum()),
        "n_lowered": int((f["price_change_pct"] < -0.5).sum()),
        "n_limited": int(f["binding_constraint"].notna().sum()),
        "avg_reputation": float(rep["reputation_score"].mean()) if len(rep) else None,
        "n_without_reputation": int(f["product_id"].nunique() - rep["product_id"].nunique()),
    }


def category_summary(f):
    """One row per category: average margins, simulated profit and gain."""
    g = f.groupby("category").agg(
        products=("product_id", "nunique"),
        margin_current=("margin_pct_current", "mean"),
        margin_recommended=("margin_pct_recommended", "mean"),
        profit_current=("profit_current", "sum"),
        gain=("margin_gain", "sum"),
    ).reset_index()
    g["gain_pct"] = g["gain"] / g["profit_current"] * 100
    return g


def fmt_big(x):
    """1234567 -> '1.2 M' (profit is in model units, so compact numbers read better)."""
    if x is None:
        return "n/a"
    for limit, suffix in ((1e9, " B"), (1e6, " M"), (1e3, " K")):
        if abs(x) >= limit:
            return f"{x / limit:,.1f}{suffix}"
    return f"{x:,.1f}"


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------

def chart_price_scatter(f):
    """Scatter of current vs recommended price (log axes) with the no-change diagonal."""
    lo = float(min(f["price_current"].min(), f["price_recommended"].min()))
    hi = float(max(f["price_current"].max(), f["price_recommended"].max()))
    log = alt.Scale(type="log", domain=[lo * 0.9, hi * 1.1])      # axes follow the data, not 1..300
    points = alt.Chart(f).mark_circle(size=60, opacity=0.75).encode(
        x=alt.X("price_current:Q", scale=log, title="Current price"),
        y=alt.Y("price_recommended:Q", scale=log, title="Recommended price"),
        color=alt.Color("price_change_pct:Q", scale=alt.Scale(scheme="redyellowgreen", domainMid=0), title="Change %"),
        tooltip=["product_id", "category", "price_current", "price_recommended", "price_change_pct"])
    diagonal = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(color="gray", strokeDash=[4, 4]).encode(
        x=alt.X("v:Q", scale=log), y=alt.Y("v:Q", scale=log))
    return (points + diagonal).properties(height=320)


def chart_change_histogram(f):
    """Histogram of the recommended price changes."""
    return alt.Chart(f).mark_bar(color="#457b9d").encode(
        x=alt.X("price_change_pct:Q", bin=alt.Bin(maxbins=30), title="Recommended price vs current (%)"),
        y=alt.Y("count():Q", title="Products")).properties(height=320)


def chart_margin_by_category(cs):
    """Grouped bars: average margin before and after, per category."""
    long = cs.melt(id_vars="category", value_vars=["margin_current", "margin_recommended"],
                   var_name="series", value_name="margin")
    long["series"] = long["series"].map({"margin_current": "Current", "margin_recommended": "Recommended"})
    return alt.Chart(long).mark_bar().encode(
        y=alt.Y("category:N", sort=alt.EncodingSortField("margin", order="descending"), title=None),
        yOffset="series:N",
        x=alt.X("margin:Q", axis=alt.Axis(format="%"), title="Average margin on price"),
        color=alt.Color("series:N", scale=alt.Scale(range=["#8d99ae", "#2a9d8f"]), title=None),
        tooltip=["category", "series", alt.Tooltip("margin:Q", format=".1%")]
    ).properties(height=max(300, 22 * len(cs)))


def chart_gain_by_category(cs):
    """Bars: simulated profit gain per category."""
    return alt.Chart(cs).mark_bar().encode(
        y=alt.Y("category:N", sort="-x", title=None),
        x=alt.X("gain_pct:Q", title="Simulated profit gain (%)"),
        color=alt.condition(alt.datum.gain_pct >= 0, alt.value("#2a9d8f"), alt.value("#e76f51")),
        tooltip=["category", alt.Tooltip("gain_pct:Q", format=".1f"), "products"]
    ).properties(height=max(300, 22 * len(cs)))


def chart_reputation(f):
    """Scatter of reputation score vs recommended price change."""
    rep = f[f["reputation_available"]]
    return alt.Chart(rep).mark_circle(size=60, opacity=0.7, color="#e9c46a").encode(
        x=alt.X("reputation_score:Q", scale=alt.Scale(zero=False), title="Reputation score (Agent 2)"),
        y=alt.Y("price_change_pct:Q", title="Recommended price change (%)"),
        tooltip=["product_id", "category", "reputation_score", "price_change_pct"]).properties(height=320)


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------

def render_overview(f):
    """Overview tab: KPI cards and charts for the filtered products."""
    k = compute_kpis(f)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products analyzed", k["n_products"])
    c2.metric("Average price (recommended)", f"{k['avg_price_recommended']:,.2f}",
              f"{k['avg_price_change_pct']:+.2f}% vs current {k['avg_price_current']:,.2f}", delta_color="off")
    c3.metric("Predicted demand (units/week)", fmt_big(k["demand_recommended"]),
              f"{(k['demand_recommended'] / k['demand_current'] - 1) * 100:+.1f}% vs current price", delta_color="off",
              help="Agent 1's LSTM forecast, adjusted for the price change with the assumed elasticity.")
    c4.metric("Average reputation score", f"{k['avg_reputation']:.2f}" if k["avg_reputation"] is not None else "n/a",
              f"{k['n_without_reputation']} products without a linked score", delta_color="off",
              help="Agent 2's Transformer-based score in [0, 1]. Products without one use a neutral value.")
    d1, d2, d3 = st.columns(3)
    d1.metric("Expected weekly profit (recommended)", fmt_big(k["profit_recommended"]),
              f"{k['gain_pct']:+.2f}% vs current {fmt_big(k['profit_current'])}", delta_color="off",
              help="(price - cost) x predicted demand, in model units. SIMULATED.")
    d2.metric("Estimated gain per week (simulated)", fmt_big(k["gain"]),
              help="Recommended profit minus current-price profit under the assumed elasticity. Not an observed result.")
    d3.metric("Prices raised / lowered", f"{k['n_raised']} / {k['n_lowered']}",
              f"{k['n_limited']} limited by a pricing rule", delta_color="off")

    cs = category_summary(f)
    left, right = st.columns(2)
    with left:
        st.subheader("Current vs recommended price")
        st.caption("Each dot is a product. Above the dashed line = price raised; below = lowered.")
        st.altair_chart(chart_price_scatter(f))
    with right:
        st.subheader("How far prices move")
        st.caption("Moves are capped by the pricing rules (for example +/-10% per update).")
        st.altair_chart(chart_change_histogram(f))
    st.subheader("Margin by category")
    st.caption("Average margin on price, before and after the recommended prices.")
    st.altair_chart(chart_margin_by_category(cs))
    left, right = st.columns(2)
    with left:
        st.subheader("Simulated gain by category")
        st.altair_chart(chart_gain_by_category(cs))
    with right:
        st.subheader("Reputation vs price change")
        st.caption("Reputation lowers the assumed price sensitivity, but cost, competitor price and demand also matter.")
        st.altair_chart(chart_reputation(f))


def render_product(df_all, f):
    """Product tab: one product before and after, with the reasoning."""
    st.subheader("One product, before and after")
    products = f.groupby("product_id").agg(category=("category", "first"), gain=("margin_gain", "sum")).reset_index()
    if products.empty:
        st.info("No product matches the current filters.")
        return
    labels = [f"{r.product_id} - {r.category}" for r in products.itertuples()]
    default = products["gain"].abs().to_numpy().argmax()          # start with the product that changes most
    choice = st.selectbox("Product", labels, index=int(default))
    pid = choice.split(" - ")[0]

    week = f["week"].iloc[0]
    row = f[(f["product_id"] == pid) & (f["week"] == week)]
    row = row.iloc[0] if len(row) else f[f["product_id"] == pid].iloc[0]
    st.caption(f"Recommendation for the week of {row['week'].date()} - category {row['category']}")

    before, after = st.columns(2)
    with before:
        st.markdown("#### Before (current price)")
        st.metric("Price", f"{row['price_current']:,.2f}")
        st.metric("Predicted demand", fmt_big(row["demand_current"]))
        st.metric("Expected weekly profit", fmt_big(row["profit_current"]))
        st.metric("Margin on price", f"{row['margin_pct_current']:.1%}")
    with after:
        st.markdown("#### After (recommended price)")
        st.metric("Price", f"{row['price_recommended']:,.2f}", f"{row['price_change_pct']:+.1f}%", delta_color="off")
        st.metric("Predicted demand", fmt_big(row["demand_recommended"]),
                  f"{(row['demand_recommended'] / row['demand_current'] - 1) * 100:+.1f}%", delta_color="off")
        st.metric("Expected weekly profit", fmt_big(row["profit_recommended"]),
                  f"{(row['profit_recommended'] / row['profit_current'] - 1) * 100:+.1f}%", delta_color="off")
        st.metric("Margin on price", f"{row['margin_pct_recommended']:.1%}",
                  f"{(row['margin_pct_recommended'] - row['margin_pct_current']) * 100:+.1f} pts", delta_color="off")

    info1, info2, info3 = st.columns(3)
    info1.metric("Competitor price", f"{row['competitor_price']:,.2f}" if pd.notna(row["competitor_price"]) else "n/a")
    info2.metric("Reputation score", f"{row['reputation_score']:.2f}" if row["reputation_available"]
                 else "n/a (neutral value used)")
    info3.metric("Assumed elasticity", f"{row['elasticity_used']:.2f}")

    st.markdown("**Why did the price change?**")
    st.info(row["reasoning"])
    if row["flags"]:
        st.caption("Flags: " + row["flags"].replace(";", ", "))

    schedule = df_all[(df_all["product_id"] == pid)].sort_values("week")
    chart_df = schedule.set_index("week")[["price_current", "price_recommended"]]
    st.markdown("**Recommended price over the forecast weeks**")
    st.line_chart(chart_df)


def render_table(f):
    """Table tab: all filtered recommendations with a CSV download."""
    st.subheader("All recommendations (filtered)")
    cols = ["product_id", "category", "week", "price_current", "price_recommended", "price_change_pct",
            "demand_current", "demand_recommended", "profit_current", "profit_recommended", "margin_gain",
            "reputation_score", "binding_constraint"]
    table = f[cols].sort_values("margin_gain", ascending=False)
    st.dataframe(table, hide_index=True)
    st.download_button("Download this table (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="recommended_prices_filtered.csv", mime="text/csv")


def render_evidence(cfg, info):
    """Evidence tab: assumptions, regression summary, backtest and data-quality notes."""
    st.subheader("What this model can and cannot tell us")
    st.markdown(
        "- **Demand** comes from Agent 1's LSTM forecast; **reputation** from Agent 2's Transformer.\n"
        "- **Price sensitivity (elasticity) is assumed**, because the sales data contains no historical prices. "
        "It is adjusted per product by competitor price gap, reputation and demand pressure.\n"
        "- The link between the 120 priced products, the 32 forecast families and the reputation scores is "
        "**synthetic** (the datasets describe different businesses).\n"
        "- All profit and gain values are **simulated estimates**, never observed results."
    )
    reg = info.get("regression")
    if reg:
        st.markdown("#### Regression (validation of the method)")
        a, b, c = st.columns(3)
        a.metric("Simulated true elasticity", reg["simulated_true_elasticity"])
        b.metric("Estimated by the regression", f"{reg['simulated_estimated_elasticity']:.3f}")
        c.metric("Real data: promotion effect R2 (last year)", f"{reg['real_data_promo_test_R2']:.2f}",
                 help="Negative = no better than predicting each family's average. The real data cannot support a price elasticity.")
        st.caption("The first two numbers come from SIMULATED data with a known answer: they show the method works, "
                   "not what real customers do.")
    reports = resolve_path(cfg["paths"]["reports_dir"])
    grid_file = reports / "backtest_elasticity_grid.csv"
    fig_dir = resolve_path(cfg["backtest"]["figures_dir"])
    st.markdown("#### Backtest (simulated)")
    if grid_file.exists() and (fig_dir / "backtest_1_gain_vs_true_elasticity.png").exists():
        st.caption("What if the real elasticity differs from the assumed one? The yellow bars show that the gain depends "
                   "on a few large products: the other products can lose.")
        st.image(str(fig_dir / "backtest_1_gain_vs_true_elasticity.png"), width=720)
        st.dataframe(pd.read_csv(grid_file).round(2), hide_index=True)
    else:
        st.info("Run `python -m src.pricing.backtest` to generate the backtest charts.")
    if info.get("warnings"):
        with st.expander("Data quality notes from the last pipeline run"):
            for w in info["warnings"]:
                st.write("- " + w)


@st.cache_data(show_spinner=False)
def load_data():
    """Read the exported recommendations once (cached until the next refresh)."""
    cfg = load_config()
    return load_recommendations(cfg), load_run_info(cfg)


def main():
    """Build the page."""
    st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
    cfg = load_config()
    st.title("Dynamic Pricing & Competition in E-Commerce")
    st.caption("Agent 3 - price optimization built on Agent 1 (LSTM demand forecast) and Agent 2 (Transformer reputation)")
    st.warning(BANNER)

    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh data and recalculate prices", type="primary"):
            with st.spinner("Running the pipeline..."):
                try:
                    result = run_pipeline(cfg)
                    load_data.clear()
                    st.success(f"Done in {result.metadata['duration_seconds']}s - {result.metadata['n_rows']} recommendations.")
                except Exception as exc:                 # show the problem instead of a stack trace
                    st.error(f"Refresh failed: {exc}")

    try:
        df, info = load_data()
    except FileNotFoundError:
        st.error("No recommendations yet. Click the button in the sidebar (or run `python -m src.pipeline.pipeline`).")
        return

    with st.sidebar:
        categories = st.multiselect("Categories", sorted(df["category"].dropna().unique()),
                                    placeholder="All categories")
        week = st.selectbox("Forecast week", [str(pd.Timestamp(w).date()) for w in sorted(df["week"].unique())])
        if info.get("generated_at"):
            st.caption(f"Last pipeline run: {info['generated_at']}")
        with st.expander("How to read this dashboard"):
            st.write("1. Overview: totals and charts for the filtered products.\n"
                     "2. Product detail: one product before and after, with the reason.\n"
                     "3. Table: every recommendation, downloadable.\n"
                     "4. Model and evidence: what is assumed and what is simulated.")

    f = filter_data(df, categories, week)
    if f.empty:
        st.info("No recommendations for the selected filters.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Product detail", "Table", "Model and evidence"])
    with tab1:
        render_overview(f)
    with tab2:
        render_product(df[df["status"] == "ok"].pipe(lambda d: d[d["category"].isin(categories)] if categories else d), f)
    with tab3:
        render_table(f)
    with tab4:
        render_evidence(cfg, info)


if __name__ == "__main__":
    main()
