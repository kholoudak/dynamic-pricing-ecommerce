
import torch
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(
        self,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        print(f"Loading Transformer model: {model_name}...")

        self.device = 0 if torch.cuda.is_available() else -1

        self.analyzer = pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=self.device
        )

        print("Transformer model loaded successfully.")

    def predict(self, texts, batch_size=16):
        """
        Analyze a list of texts.
        Returns positive, neutral and negative probabilities.
        """

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            predictions = self.analyzer(
                batch,
                batch_size=batch_size,
                truncation=True,
                max_length=256
            )

            for prediction in predictions:
                scores = {
                    item["label"].lower(): float(item["score"])
                    for item in prediction
                }

                results.append({
                    "positive": scores.get("positive", 0.0),
                    "neutral": scores.get("neutral", 0.0),
                    "negative": scores.get("negative", 0.0)
                })

        return results
