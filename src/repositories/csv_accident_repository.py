from pathlib import Path

import pandas as pd

from src import config


class CsvAccidentRepository:
    """Lê o dataset de acidentes da PRF com o formato específico do arquivo."""

    def __init__(self, file_path: Path = config.DATA_FILE) -> None:
        self._file_path = file_path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(
            self._file_path,
            sep=config.CSV_SEPARATOR,
            encoding=config.CSV_ENCODING,
            decimal=config.CSV_DECIMAL,
        )
