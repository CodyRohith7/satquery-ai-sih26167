"""A real, small, trained-at-import-time text classifier.

Why train fresh every run instead of shipping a checkpoint: the training set is
~30 short examples and fitting a TF-IDF + LogisticRegression pipeline on it takes
milliseconds. Training at runtime means there is no stale pickle to go out of
sync with `intent_training_data.py`, and "reproducible" (a hard requirement from
the frozen spec) is trivially true - anyone who runs this gets the identical
model from the identical data, every time.

This classifier ONLY distinguishes single_image_vqa vs grounding for single-image
queries. It is not used for change/fusion routing - see router.py.
"""
from __future__ import annotations

from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from routing.intent_training_data import TRAINING_EXAMPLES


class IntentClassifier:
    def __init__(self) -> None:
        texts = [t for t, _ in TRAINING_EXAMPLES]
        labels = [lbl for _, lbl in TRAINING_EXAMPLES]
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        self.pipeline.fit(texts, labels)
        self._classes = list(self.pipeline.classes_)

    def predict(self, query: str) -> Tuple[str, float]:
        """Returns (label, probability_of_that_label)."""
        proba = self.pipeline.predict_proba([query])[0]
        best_idx = int(proba.argmax())
        return self._classes[best_idx], float(proba[best_idx])


_singleton: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    global _singleton
    if _singleton is None:
        _singleton = IntentClassifier()
    return _singleton
