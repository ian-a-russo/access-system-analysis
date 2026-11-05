import pandas as pd

from src import config
from src.services.data_cleaner import AccidentDataCleaner


def build_cleaner() -> AccidentDataCleaner:
    return AccidentDataCleaner(
        config.TARGET, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    )


def test_clean_extracts_hour_and_month(raw_accidents: pd.DataFrame) -> None:
    cleaned = build_cleaner().clean(raw_accidents)

    assert cleaned["hour"].iloc[0] == 8
    assert cleaned["month"].iloc[0] == 3


def test_clean_drops_rows_without_target(raw_accidents: pd.DataFrame) -> None:
    raw_accidents.loc[0, config.TARGET] = None

    cleaned = build_cleaner().clean(raw_accidents)

    assert len(cleaned) == len(raw_accidents) - 1


def test_clean_turns_invalid_numbers_into_nan(raw_accidents: pd.DataFrame) -> None:
    raw_accidents["km"] = raw_accidents["km"].astype(object)
    raw_accidents.loc[1, "km"] = "abc"

    cleaned = build_cleaner().clean(raw_accidents)

    assert pd.isna(cleaned["km"].iloc[1])


def test_clean_converts_categorical_columns_to_text(raw_accidents: pd.DataFrame) -> None:
    cleaned = build_cleaner().clean(raw_accidents)

    assert cleaned["br"].iloc[0] == "116"


def test_clean_does_not_mutate_input(raw_accidents: pd.DataFrame) -> None:
    original_columns = list(raw_accidents.columns)

    build_cleaner().clean(raw_accidents)

    assert list(raw_accidents.columns) == original_columns
