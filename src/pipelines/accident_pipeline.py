import logging

import pandas as pd

from src import config
from src.repositories.csv_accident_repository import CsvAccidentRepository
from src.services.accident_classifier import AccidentClassifier
from src.services.data_cleaner import AccidentDataCleaner
from src.services.exploratory_analysis import ExploratoryAnalysis

logger = logging.getLogger(__name__)


class AccidentPipeline:
    """Orquestra o fluxo: carga -> limpeza -> EDA -> treino -> avaliação."""

    def __init__(
        self,
        repository: CsvAccidentRepository,
        cleaner: AccidentDataCleaner,
        exploration: ExploratoryAnalysis,
        classifier: AccidentClassifier,
    ) -> None:
        self._repository = repository
        self._cleaner = cleaner
        self._exploration = exploration
        self._classifier = classifier

    def run(self, explore: bool = True, cross_validate: bool = True) -> None:
        accidents = self._cleaner.clean(self._repository.load())
        logger.info("Registros após a limpeza: %d", len(accidents))
        if explore:
            self._exploration.run(accidents)
        self._evaluate_holdout(accidents)
        if cross_validate:
            self._evaluate_cross_validation(accidents)

    def _evaluate_holdout(self, accidents: pd.DataFrame) -> None:
        result = self._classifier.fit_and_evaluate(accidents)
        logger.info("Acurácia (hold-out): %.4f", result.accuracy)
        logger.info("Classification report:\n%s", result.report)
        logger.info("Matriz de confusão:\n%s", result.confusion)

    def _evaluate_cross_validation(self, accidents: pd.DataFrame) -> None:
        scores = self._classifier.cross_validate(accidents, config.CROSS_VALIDATION_FOLDS)
        logger.info("Validação cruzada: %.4f ± %.4f", scores.mean(), scores.std())
