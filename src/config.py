from pathlib import Path

DATA_FILE = Path("data/raw/datatran2024.csv")
FIGURES_DIR = Path("docs/figures")

# Formato do arquivo da PRF: separador ';', texto em latin-1 e decimal com vírgula
CSV_SEPARATOR = ";"
CSV_ENCODING = "latin-1"
CSV_DECIMAL = ","

TARGET = "classificacao_acidente"

# Colunas como mortos, feridos_leves, feridos_graves, feridos, ilesos e ignorados
# ficam de fora de propósito: elas definem a classe do alvo (vazamento de dados).
# Regional, delegacia e uop também ficam de fora por serem redundantes com uf/município.
NUMERIC_FEATURES = ["km", "pessoas", "veiculos", "latitude", "longitude", "hour", "month"]
CATEGORICAL_FEATURES = [
    "dia_semana",
    "uf",
    "br",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]

DISTRIBUTION_COLUMNS = ["pessoas", "veiculos", "mortos", "feridos_leves", "feridos_graves"]
BAR_COLUMNS = ["tipo_acidente", "fase_dia", "condicao_metereologica", "tipo_pista"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5
