from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Categorias raras (ex.: municípios pequenos) são agrupadas para limitar a dimensionalidade
MIN_CATEGORY_FREQUENCY = 30


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    report: str
    confusion: np.ndarray


class AccidentClassifier:
    """Random Forest que prevê a gravidade do acidente.

    Pré-processamento e modelo ficam no mesmo Pipeline: na validação cruzada
    cada fold ajusta imputer e encoder apenas com os seus próprios dados de treino.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        target: str,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self._features = numeric_columns + categorical_columns
        self._target = target
        self._random_state = random_state
        self._test_size = test_size
        self._pipeline = self._build_pipeline(numeric_columns, categorical_columns)

    def fit_and_evaluate(self, accidents: pd.DataFrame) -> EvaluationResult:
        x_train, x_test, y_train, y_test = train_test_split(
            accidents[self._features],
            accidents[self._target],
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=accidents[self._target],
        )
        self._pipeline.fit(x_train, y_train)
        return self._evaluate(x_test, y_test)

    def cross_validate(self, accidents: pd.DataFrame, folds: int = 5) -> np.ndarray:
        scores = cross_val_score(
            self._pipeline,
            accidents[self._features],
            accidents[self._target],
            cv=folds,
            scoring="accuracy",
            n_jobs=-1,
        )
        return np.asarray(scores, dtype=float)

    def predict(self, accidents: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._pipeline.predict(accidents[self._features]), dtype=object)

    def _evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
        predicted = self._pipeline.predict(x_test)
        return EvaluationResult(
            accuracy=float(accuracy_score(y_test, predicted)),
            report=str(classification_report(y_test, predicted, zero_division=0)),
            confusion=confusion_matrix(y_test, predicted),
        )

    def _build_pipeline(
        self, numeric_columns: list[str], categorical_columns: list[str]
    ) -> Pipeline:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQUENCY
        )
        preprocessing = ColumnTransformer(
            [
                ("numeric", SimpleImputer(strategy="median"), numeric_columns),
                ("categorical", encoder, categorical_columns),
            ]
        )
        # "balanced" compensa o desbalanceamento: acidentes fatais são só ~7% dos registros
        forest = RandomForestClassifier(
            class_weight="balanced", random_state=self._random_state, n_jobs=-1
        )
        return Pipeline([("preprocessing", preprocessing), ("classifier", forest)])
