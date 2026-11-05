import pandas as pd
import pytest

from src import config


@pytest.fixture
def raw_accidents() -> pd.DataFrame:
    """Amostra sintética no mesmo formato do CSV da PRF (já lido com decimal=',')."""
    rows = 60
    severities = ["Sem Vítimas", "Com Vítimas Feridas", "Com Vítimas Fatais"] * (rows // 3)
    return pd.DataFrame(
        {
            "data_inversa": ["15/03/2024"] * rows,
            "horario": ["08:30:00"] * rows,
            "dia_semana": ["segunda-feira"] * rows,
            "uf": ["SP", "MG"] * (rows // 2),
            "br": [116, 381] * (rows // 2),
            "km": [10.5] * rows,
            "municipio": ["SAO PAULO"] * rows,
            "causa_acidente": ["Falta de Atenção"] * rows,
            "tipo_acidente": ["Colisão traseira", "Saída de leito carroçável"] * (rows // 2),
            "fase_dia": ["Pleno dia"] * rows,
            "sentido_via": ["Crescente"] * rows,
            "condicao_metereologica": ["Céu Claro"] * rows,
            "tipo_pista": ["Simples"] * rows,
            "tracado_via": ["Reta"] * rows,
            "uso_solo": ["Sim"] * rows,
            "pessoas": [1, 2, 3] * (rows // 3),
            "veiculos": [1, 2] * (rows // 2),
            "latitude": [-23.5] * rows,
            "longitude": [-46.6] * rows,
            config.TARGET: severities,
        }
    )
