
import pandas as pd


def calculate_reputation(df, predictions):
    """
    Calculate a reputation score for each product.

    Score:
        positive = 1
        neutral  = 0.5
        negative = 0

    Final score is the average sentiment probability.
    """

    if len(df) != len(predictions):
        raise ValueError(
            "The number of predictions must match the number of reviews."
        )

    results = df[["product_name"]].copy()

    results["positive"] = [
        prediction["positive"] for prediction in predictions
    ]

    results["neutral"] = [
        prediction["neutral"] for prediction in predictions
    ]

    results["negative"] = [
        prediction["negative"] for prediction in predictions
    ]

    # Reputation score for each review, between 0 and 1
    results["review_score"] = (
        results["positive"]
        + 0.5 * results["neutral"]
    )

    # Aggregate reviews by product
    reputation = results.groupby("product_name").agg(
        reputation_score=("review_score", "mean"),
        n_reviews=("review_score", "count"),
        positive_ratio=("positive", "mean"),
        neutral_ratio=("neutral", "mean"),
        negative_ratio=("negative", "mean")
    ).reset_index()

    # Round scores for readability
    score_columns = [
        "reputation_score",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio"
    ]

    reputation[score_columns] = reputation[score_columns].round(4)

    # Sort from highest to lowest reputation
    reputation = reputation.sort_values(
        by="reputation_score",
        ascending=False
    ).reset_index(drop=True)

    return reputation
