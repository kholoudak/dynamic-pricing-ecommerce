
import unittest
import pandas as pd

from src.nlp.preprocessing import clean_reviews
from src.nlp.reputation import calculate_reputation


class TestPreprocessing(unittest.TestCase):

    def test_clean_reviews(self):
        data = {
            "product_name": ["Phone A", "Phone B", "Phone C"],
            "product_price": [1000, 2000, 3000],
            "Rate": [5, 4, "invalid"],
            "Review": ["Excellent product", None, ""],
            "Summary": ["Good", "Very good", "No review"],
            "Sentiment": ["positive", "positive", "neutral"]
        }

        df = pd.DataFrame(data)
        result = clean_reviews(df)

        self.assertEqual(len(result), 3)
        self.assertIn("review_text", result.columns)
        self.assertEqual(result.iloc[1]["review_text"], "Very good")
        self.assertTrue(pd.isna(result.iloc[2]["Rate"]))


class TestReputation(unittest.TestCase):

    def test_calculate_reputation(self):
        df = pd.DataFrame({
            "product_name": ["Phone A", "Phone A", "Phone B"]
        })

        predictions = [
            {"positive": 1.0, "neutral": 0.0, "negative": 0.0},
            {"positive": 0.0, "neutral": 0.0, "negative": 1.0},
            {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        ]

        result = calculate_reputation(df, predictions)

        phone_a = result[result["product_name"] == "Phone A"].iloc[0]
        phone_b = result[result["product_name"] == "Phone B"].iloc[0]

        self.assertAlmostEqual(phone_a["reputation_score"], 0.5)
        self.assertAlmostEqual(phone_b["reputation_score"], 0.5)
        self.assertEqual(phone_a["n_reviews"], 2)
        self.assertEqual(phone_b["n_reviews"], 1)

    def test_invalid_prediction_count(self):
        df = pd.DataFrame({
            "product_name": ["Phone A", "Phone B"]
        })

        predictions = [
            {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
        ]

        with self.assertRaises(ValueError):
            calculate_reputation(df, predictions)


if __name__ == "__main__":
    unittest.main()
