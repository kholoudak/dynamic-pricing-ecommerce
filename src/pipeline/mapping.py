"""Build (or load) the product mapping that links the three datasets.

WHY THIS EXISTS
- Agent 1 forecasts demand per product FAMILY (32 grocery/retail families).
- The pricing file has 120 numbered products with no category.
- Agent 2 scores reputation per product NAME (Amazon/Flipkart electronics).
No real key links them, so we CREATE a transparent, deterministic mapping and save it as
``product_mapping.csv``. Anyone can open that file, see the links, and replace it with real ones.

Columns of the mapping file: product_id, category, reputation_name, mapping_type
"""
import numpy as np
import pandas as pd

from src.pipeline.loaders import DataValidationError
from src.pricing.config import resolve_path

MAPPING_COLUMNS = ["product_id", "category", "reputation_name", "mapping_type"]


def build_mapping(products, forecast, reputation, seed=42, type_label="synthetic"):
    """Create the mapping from scratch.

    - category: products are dealt to the families in round-robin order
      (product 1 -> family 1, product 2 -> family 2, ... then it wraps around).
    - reputation_name: a seeded random choice of products receives the reputation names
      (sorted alphabetically first). Products left over get no reputation (NaN).
    """
    families = sorted(forecast["product_id"].unique())
    ids = products["product_id"].tolist()          # keep the order of the file

    mapping = pd.DataFrame({"product_id": ids})
    mapping["category"] = [families[i % len(families)] for i in range(len(ids))]

    names = sorted(reputation["product_name"].unique())
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(ids))[: len(names)]   # positions of products that get a name
    mapping["reputation_name"] = np.nan
    mapping["reputation_name"] = mapping["reputation_name"].astype(object)
    for position, name in zip(chosen, names):
        mapping.at[int(position), "reputation_name"] = name

    mapping["mapping_type"] = type_label
    return mapping[MAPPING_COLUMNS]


def validate_mapping(mapping, products, forecast):
    """Raise DataValidationError if the mapping cannot be used with the other files."""
    missing_cols = [c for c in MAPPING_COLUMNS if c not in mapping.columns]
    if missing_cols:
        raise DataValidationError(f"mapping: missing columns {missing_cols}")
    if mapping["product_id"].duplicated().any():
        raise DataValidationError("mapping: duplicated product_id")
    no_map = set(products["product_id"]) - set(mapping["product_id"])
    if no_map:
        raise DataValidationError(f"mapping: {len(no_map)} products have no mapping, e.g. {sorted(no_map)[:3]}")
    unknown = set(mapping["category"]) - set(forecast["product_id"])
    if unknown:
        raise DataValidationError(f"mapping: categories without a forecast: {sorted(unknown)[:3]}")


def get_mapping(cfg, products, forecast, reputation, rebuild=False):
    """Load ``product_mapping.csv`` if it exists, otherwise build it and save it.

    Returns (mapping, message). Use ``rebuild=True`` to overwrite the file.
    """
    path = resolve_path(cfg["paths"]["mapping_file"])
    if path.exists() and not rebuild:
        mapping = pd.read_csv(path, dtype={"product_id": str})
        message = f"mapping: loaded existing file {path.name} ({len(mapping)} rows)."
    else:
        mapping = build_mapping(products, forecast, reputation,
                                seed=cfg["mapping"]["seed"], type_label=cfg["mapping"]["type_label"])
        path.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(path, index=False)
        message = f"mapping: built a new {cfg['mapping']['type_label']} mapping and saved it to {path.name}."
    mapping["product_id"] = mapping["product_id"].astype(str).str.strip()
    validate_mapping(mapping, products, forecast)
    return mapping, message
