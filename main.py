import argparse
import logging

from src import config
from src.pipelines.accident_pipeline import AccidentPipeline
from src.repositories.csv_accident_repository import CsvAccidentRepository
from src.services.accident_classifier import AccidentClassifier
from src.services.data_cleaner import AccidentDataCleaner
from src.services.exploratory_analysis import ExploratoryAnalysis
from src.services.graphic_generator import GraphicGenerator


def build_pipeline() -> AccidentPipeline:
    return AccidentPipeline(
        repository=CsvAccidentRepository(),
        cleaner=AccidentDataCleaner(
            config.TARGET, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
        ),
        exploration=ExploratoryAnalysis(
            GraphicGenerator(config.FIGURES_DIR),
            config.DISTRIBUTION_COLUMNS,
            config.BAR_COLUMNS,
        ),
        classifier=AccidentClassifier(
            config.NUMERIC_FEATURES,
            config.CATEGORICAL_FEATURES,
            config.TARGET,
            config.RANDOM_STATE,
            config.TEST_SIZE,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classificação da gravidade de acidentes (PRF)")
    parser.add_argument("--skip-eda", action="store_true", help="não gera os gráficos")
    parser.add_argument("--skip-cv", action="store_true", help="não roda a validação cruzada")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_pipeline().run(explore=not args.skip_eda, cross_validate=not args.skip_cv)


if __name__ == "__main__":
    main()
