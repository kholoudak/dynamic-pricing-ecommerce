
from pathlib import Path

import pandas as pd

from src.nlp.preprocessing import load_reviews, clean_reviews
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.reputation import calculate_reputation


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset-SA.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_PATH = OUTPUT_DIR / "reputation_checkpoint.csv"
OUTPUT_PATH = OUTPUT_DIR / "reputation_scores.csv"

# Number of reviews processed before each checkpoint
CHUNK_SIZE = 1000


def main():

    print("=" * 60)
    print("           PRODUCT REPUTATION AGENT")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")

    df = load_reviews(DATA_PATH)
    print(f"Dataset loaded: {len(df)} reviews")

    # 2. Preprocessing
    print("\n[2/5] Cleaning and preprocessing...")

    df = clean_reviews(df)

    total_reviews = len(df)

    print(f"Usable reviews: {total_reviews}")
    print(f"Unique products: {df['product_name'].nunique()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Check existing progress
    print("\n[3/5] Checking saved progress...")

    if CHECKPOINT_PATH.exists():

        checkpoint = pd.read_csv(CHECKPOINT_PATH)

        required_columns = [
            "product_name",
            "positive",
            "neutral",
            "negative"
        ]

        if not all(col in checkpoint.columns for col in required_columns):
            raise ValueError(
                "Invalid checkpoint. Delete reputation_checkpoint.csv "
                "and restart."
            )

        completed = len(checkpoint)

        if completed > total_reviews:
            raise ValueError(
                "Checkpoint contains more reviews than the current dataset. "
                "Delete reputation_checkpoint.csv and restart."
            )

        expected_names = (
            df["product_name"]
            .iloc[:completed]
            .reset_index(drop=True)
            .astype(str)
        )

        saved_names = (
            checkpoint["product_name"]
            .reset_index(drop=True)
            .astype(str)
        )

        if not expected_names.equals(saved_names):
            raise ValueError(
                "Checkpoint does not match the current dataset. "
                "Delete reputation_checkpoint.csv and restart."
            )

        print(f"Previous progress found: {completed}/{total_reviews}")

    else:
        checkpoint = None
        completed = 0
        print("No previous progress found. Starting from zero.")

    # 4. Analyze reviews in chunks
    print("\n[4/5] Analyzing customer reviews...")

    if completed < total_reviews:

        analyzer = SentimentAnalyzer()

        for start in range(completed, total_reviews, CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, total_reviews)

            chunk = df.iloc[start:end]
            texts = chunk["review_text"].tolist()

            predictions = analyzer.predict(
                texts,
                batch_size=16
            )

            chunk_results = pd.DataFrame({
                "product_name": chunk["product_name"].tolist(),
                "positive": [p["positive"] for p in predictions],
                "neutral": [p["neutral"] for p in predictions],
                "negative": [p["negative"] for p in predictions]
            })

            chunk_results.to_csv(
                CHECKPOINT_PATH,
                mode="a",
                header=not CHECKPOINT_PATH.exists(),
                index=False,
                encoding="utf-8"
            )

            print(
                f"Progress: {end}/{total_reviews} reviews "
                f"({end / total_reviews * 100:.1f}%)"
            )

    else:
        print("All reviews have already been analyzed.")

    # 5. Calculate product reputation
    print("\n[5/5] Calculating product reputation...")

    checkpoint = pd.read_csv(CHECKPOINT_PATH)

    predictions = checkpoint[
        ["positive", "neutral", "negative"]
    ].to_dict(orient="records")

    reputation = calculate_reputation(df, predictions)

    reputation.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    # Final report
    print("\n" + "=" * 60)
    print("       AGENT 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTotal reviews analyzed: {len(predictions)}")
    print(f"Total products analyzed: {len(reputation)}")

    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Final output: {OUTPUT_PATH}")

    print("\nTop 5 products:")
    print(reputation.head(5).to_string(index=False))

    print("\nBottom 5 products:")
    print(reputation.tail(5).to_string(index=False))

    print("\nReputation statistics:")
    print(reputation["reputation_score"].describe())

    print("\nExecution finished.")


if __name__ == "__main__":
    main()
