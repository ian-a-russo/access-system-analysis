import logging
from collections.abc import Callable

import pandas as pd

from src.services.graphic_generator import GraphicGenerator

logger = logging.getLogger(__name__)


class ExploratoryAnalysis:
    """Resume o dataset e gera os gráficos exploratórios."""

    def __init__(
        self,
        graphics: GraphicGenerator,
        distribution_columns: list[str],
        bar_columns: list[str],
        include_heatmap: bool = False,
    ) -> None:
        self._graphics = graphics
        self._distribution_columns = distribution_columns
        self._bar_columns = bar_columns
        self._include_heatmap = include_heatmap

    def run(self, accidents: pd.DataFrame) -> None:
        logger.info("Resumo estatístico:\n%s", accidents.describe(include="all"))
        plot_distribution = self._graphics.plot_distribution
        self._plot_available(accidents, self._distribution_columns, plot_distribution)
        self._plot_available(accidents, self._bar_columns, self._graphics.bar_plot)
        if self._include_heatmap:
            self._graphics.correlation_heatmap(accidents)

    def _plot_available(
        self,
        accidents: pd.DataFrame,
        columns: list[str],
        plot: Callable[[pd.DataFrame, str], None],
    ) -> None:
        available = [column for column in columns if column in accidents.columns]
        for column in available:
            plot(accidents, column)
