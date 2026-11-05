import pandas as pd

from src import config
from src.services.accident_classifier import AccidentClassifier
from src.services.data_cleaner import AccidentDataCleaner


def build_classifier() -> AccidentClassifier:
    return AccidentClassifier(
        config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES, config.TARGET
    )


def clean(raw_accidents: pd.DataFrame) -> pd.DataFrame:
    return AccidentDataCleaner(
        config.TARGET, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    ).clean(raw_accidents)


def test_fit_and_evaluate_reports_accuracy_between_zero_and_one(
    raw_accidents: pd.DataFrame,
) -> None:
    result = build_classifier().fit_and_evaluate(clean(raw_accidents))

    assert 0.0 <= result.accuracy <= 1.0
    assert result.confusion.shape == (3, 3)


def test_cross_validate_returns_one_score_per_fold(raw_accidents: pd.DataFrame) -> None:
    scores = build_classifier().cross_validate(clean(raw_accidents), folds=3)

    assert len(scores) == 3


def test_predict_returns_known_severity_labels(raw_accidents: pd.DataFrame) -> None:
    classifier = build_classifier()
    accidents = clean(raw_accidents)
    classifier.fit_and_evaluate(accidents)

    predictions = classifier.predict(accidents.head(5))

    assert set(predictions) <= set(accidents[config.TARGET])
