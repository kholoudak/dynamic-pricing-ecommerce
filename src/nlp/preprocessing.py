
import pandas as pd


def load_reviews(file_path):
    """Load and validate the Flipkart reviews dataset."""

    df = pd.read_csv(file_path, low_memory=False)

    required_columns = [
        "product_name",
        "product_price",
        "Rate",
        "Review",
        "Summary",
        "Sentiment"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def clean_reviews(df):
    """Clean reviews and prepare text for sentiment analysis."""

    df = df.copy()

    # Remove rows without a product name
    df["product_name"] = df["product_name"].astype("string").str.strip()

    df = df.dropna(subset=["product_name"])
    df = df[df["product_name"] != ""]

    # Convert ratings to numeric values
    # Invalid values (including the 3 malformed entries) become NaN
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")

    # Ratings must be between 1 and 5
    df.loc[~df["Rate"].between(1, 5), "Rate"] = float("nan")

    # Clean review and summary text
    df["Review"] = df["Review"].fillna("").astype(str).str.strip()
    df["Summary"] = df["Summary"].fillna("").astype(str).str.strip()

    # Use Review when available, otherwise use Summary
    df["review_text"] = df["Review"].where(
        df["Review"] != "",
        df["Summary"]
    )

    # Remove rows without any usable text
    df = df[df["review_text"] != ""]

    # Normalize existing sentiment labels
    df["Sentiment"] = (
        df["Sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_sentiments = ["positive", "neutral", "negative"]
    df = df[df["Sentiment"].isin(valid_sentiments)]

    # Remove unnecessary columns for the NLP pipeline
    df = df[
        [
            "product_name",
            "product_price",
            "Rate",
            "review_text",
            "Sentiment"
        ]
    ]

    return df.reset_index(drop=True)
