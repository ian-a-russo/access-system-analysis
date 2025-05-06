import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataPreprocessing:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_col: str,
        numeric_cols: list[str],
        categorical_cols: list[str]
    ):
        self.dataframe = dataframe.copy()
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        self.dataframe['data_inversa'] = pd.to_datetime(
            self.dataframe['data_inversa'], format='%d/%m/%Y', dayfirst=True, errors='coerce'
        )
        self.dataframe['horario'] = pd.to_timedelta(self.dataframe['horario'], errors='coerce')



        for col in self.numeric_cols:
            self.dataframe[col] = pd.to_numeric(
                self.dataframe[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.preprocessor = ColumnTransformer([
            ('num', num_pipe, self.numeric_cols),
            ('cat', cat_pipe, self.categorical_cols)
        ], remainder='drop')

    def fit_transform(self):
        print('Pré-processando os dados...')

        self.dataframe = self.dataframe.dropna(subset=['data_inversa', 'horario', self.target_col])

        if self.dataframe.empty:
            raise ValueError("O dataframe ficou vazio após o dropna em 'data_inversa' e 'horario'.")

        X = self.dataframe.drop(columns=[self.target_col])
        y = self.dataframe[self.target_col]

        print(f"Shape de X: {X.shape}")
        print(X.head())

        X_processed = self.preprocessor.fit_transform(X)

        return X_processed, y



    def transform(self, dataframe_new: pd.DataFrame):
        print('Transformando os dados...')

        dataframe2 = dataframe_new.copy()

        dataframe2['data_inversa'] = pd.to_datetime(
            dataframe2['data_inversa'], format='%d/%m/%Y', dayfirst=True, errors='coerce'
        )
        dataframe2['horario'] = pd.to_timedelta(dataframe2['horario'] + ':00', errors='coerce')
        for col in self.numeric_cols:
            dataframe2[col] = pd.to_numeric(
                dataframe2[col].astype(str).str.replace(',', '.'), errors='coerce'
            )
        X_new = dataframe2.drop(columns=[self.target_col], errors='ignore')
        return self.preprocessor.transform(X_new)


        