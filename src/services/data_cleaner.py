import pandas as pd


class AccidentDataCleaner:
    """Padroniza tipos e cria atributos de tempo. Não aprende nada dos dados.

    O aprendizado (imputação, encoding) fica no Pipeline do modelo, para ser
    ajustado só com o treino e evitar vazamento para a validação.
    """

    def __init__(
        self,
        target: str,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self._target = target
        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns

    def clean(self, accidents: pd.DataFrame) -> pd.DataFrame:
        cleaned = accidents.dropna(subset=[self._target]).copy()
        cleaned["hour"] = self._extract_hour(cleaned["horario"])
        cleaned["month"] = self._extract_month(cleaned["data_inversa"])
        self._coerce_numeric(cleaned)
        self._coerce_categorical(cleaned)
        return cleaned

    def _extract_hour(self, times: pd.Series) -> pd.Series:
        return pd.to_datetime(times, format="%H:%M:%S", errors="coerce").dt.hour

    def _extract_month(self, dates: pd.Series) -> pd.Series:
        return pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce").dt.month

    def _coerce_numeric(self, accidents: pd.DataFrame) -> None:
        for column in self._numeric_columns:
            accidents[column] = pd.to_numeric(accidents[column], errors="coerce")

    def _coerce_categorical(self, accidents: pd.DataFrame) -> None:
        # Valores ausentes viram a categoria "nan", tratada como mais uma classe
        for column in self._categorical_columns:
            accidents[column] = accidents[column].astype(str)
