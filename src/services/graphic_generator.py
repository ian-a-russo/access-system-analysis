from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

FIGURE_SIZE = (6, 4)


class GraphicGenerator:
    """Gera gráficos de EDA e os salva como PNG.

    Usa a API orientada a objetos do matplotlib (Figure) em vez do pyplot,
    evitando estado global e dependência de backend gráfico.
    """

    def __init__(self, save_dir: Path) -> None:
        self._save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    def plot_distribution(self, accidents: pd.DataFrame, column: str, bins: int = 30) -> None:
        fig = Figure(figsize=FIGURE_SIZE)
        axes = fig.subplots()
        axes.hist(accidents[column].dropna(), bins=bins, edgecolor="black")
        self._decorate(fig, f"Distribuição de {column}", column, "Frequência")
        self._save(fig, f"distribution_{column}.png")

    def bar_plot(self, accidents: pd.DataFrame, column: str) -> None:
        fig = Figure(figsize=FIGURE_SIZE)
        axes = fig.subplots()
        accidents[column].value_counts().plot(kind="bar", edgecolor="black", ax=axes)
        self._decorate(fig, f"Distribuição de {column}", column, "Frequência")
        self._save(fig, f"bar_{column}.png")

    def correlation_heatmap(self, accidents: pd.DataFrame) -> None:
        correlation = accidents.select_dtypes(include="number").corr()
        fig = Figure(figsize=(8, 6))
        axes = fig.subplots()
        image = axes.imshow(correlation, interpolation="nearest")
        fig.colorbar(image, ax=axes)
        axes.set_xticks(range(len(correlation)), correlation.columns, rotation=90)
        axes.set_yticks(range(len(correlation)), correlation.columns)
        axes.set_title("Heatmap de Correlação")
        fig.tight_layout()
        self._save(fig, "heatmap.png")

    def _decorate(self, fig: Figure, title: str, xlabel: str, ylabel: str) -> None:
        axes = fig.axes[0]
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        fig.tight_layout()

    def _save(self, fig: Figure, filename: str) -> None:
        fig.savefig(self._save_dir / filename)
