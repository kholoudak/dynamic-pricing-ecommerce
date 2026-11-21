"""
Step 1 - Dataset inspection / Etape 1 - Inspection du jeu de donnees.

Usage (Windows 11, PowerShell):
    python inspect_dataset.py "C:\\path\\to\\your_file.csv"

The script assumes NOTHING about column names: it only reports what exists.
Le script ne suppose AUCUN nom de colonne : il rapporte uniquement ce qui existe.
"""
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV, trying common encodings. Returns a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file with utf-8 / utf-8-sig / latin-1")


def guess_date_columns(df: pd.DataFrame) -> list:
    """Return columns that look like dates (name hint + successful parsing)."""
    found = []
    for col in df.columns:
        hint = any(k in col.lower() for k in ("date", "time", "day", "week", "month"))
        if df[col].dtype == object or hint:
            sample = df[col].dropna().astype(str).head(200)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                found.append(col)
    return found


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    df = load_csv(Path(sys.argv[1]))

    print("=" * 70, "\nSHAPE (rows, columns):", df.shape)
    print("\nCOLUMNS + DTYPES + NULLS + UNIQUE VALUES")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    print(summary)

    print("\nFIRST 5 ROWS\n", df.head())
    print("\nLAST 5 ROWS\n", df.tail())
    print("\nNUMERIC STATISTICS\n", df.describe().T)
    print("\nFULL DUPLICATE ROWS:", int(df.duplicated().sum()))

    date_cols = guess_date_columns(df)
    print("\nPOSSIBLE DATE COLUMNS:", date_cols or "none detected")
    for col in date_cols:
        d = pd.to_datetime(df[col], errors="coerce")
        print(f"  {col}: {d.min()} -> {d.max()} | unparsable: {int(d.isna().sum())}")

    cat_cols = [c for c in df.columns if df[c].dtype == object and c not in date_cols]
    for col in cat_cols[:8]:
        print(f"\nTop values of '{col}' ({df[col].nunique()} unique):")
        print(df[col].value_counts().head(10))


if __name__ == "__main__":
    main()
