from pathlib import Path

import pandas as pd

from src.repositories.csv_accident_repository import CsvAccidentRepository
from src.services.exploratory_analysis import ExploratoryAnalysis
from src.services.graphic_generator import GraphicGenerator


def test_repository_reads_semicolon_latin1_file_with_decimal_comma(tmp_path: Path) -> None:
    csv_file = tmp_path / "acidentes.csv"
    csv_file.write_bytes("km;municipio\n10,5;SÃO PAULO\n".encode("latin-1"))

    accidents = CsvAccidentRepository(csv_file).load()

    assert accidents["km"].iloc[0] == 10.5
    assert accidents["municipio"].iloc[0] == "SÃO PAULO"


def test_exploratory_analysis_saves_figures_for_available_columns(
    tmp_path: Path, raw_accidents: pd.DataFrame
) -> None:
    analysis = ExploratoryAnalysis(
        GraphicGenerator(tmp_path),
        distribution_columns=["pessoas", "coluna_inexistente"],
        bar_columns=["fase_dia"],
        include_heatmap=True,
    )

    analysis.run(raw_accidents)

    generated = sorted(file.name for file in tmp_path.iterdir())
    assert generated == ["bar_fase_dia.png", "distribution_pessoas.png", "heatmap.png"]
