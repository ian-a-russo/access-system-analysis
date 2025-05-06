import pandas as pd
from src.services.data_preprocessing import DataPreprocessing
from src.services.decision_tree_pipeline import DecisionTreePipeline
from src.services.graphic_generator import GraphicGenerator

class AccidentPipeline:
    def __init__(self, dataframe: pd.DataFrame):
        self.preprocessor = DataPreprocessing(
            dataframe=dataframe,
            target_col='classificacao_acidente',
            numeric_cols=[
                'km','pessoas','mortos','feridos_leves','feridos_graves',
                'ilesos','ignorados','feridos','veiculos','latitude','longitude'
            ],
            categorical_cols=[c for c in dataframe.columns if c not in [
                'id','data_inversa','classificacao_acidente',
                'km','pessoas','mortos','feridos_leves','feridos_graves',
                'ilesos','ignorados','feridos','veiculos','latitude','longitude'
            ]]
        )
        
        self.dataframe = self.preprocessor.dataframe
        self.target = 'classificacao_acidente'
        self.numeric_cols = self.preprocessor.numeric_cols
        self.categorical_cols = self.preprocessor.categorical_cols
        self.model = DecisionTreePipeline()
        self.graphics = GraphicGenerator('src/data/graphics')

    def run(self):
        X, y = self.preprocessor.fit_transform()
        self.model.train(X, y)
        self.model.evaluate()
        self.model.cross_validate(X, y)

    def show_eda(
        self,
        dist_cols: list[str] = None,
        do_heatmap: bool = True,
        cat_cols: list[str] = None
    ):
        print('Mostrando gráficos...')
        print(self.dataframe.describe(include='all'))
        if dist_cols is None:
                dist_cols = ['km', 'pessoas', 'veiculos']
        for col in dist_cols:
            if col in self.preprocessor.numeric_cols:
                self.graphics.plot_distribution(self.dataframe, col)

        if do_heatmap:
            self.graphics.correlation_heatmap(self.dataframe)

        if cat_cols is None:
            cat_cols = ['tipo_acidente', 'fase_dia', 'condicao_metereologica', 'tipo_pista']
        for cat in cat_cols:
            if cat in self.dataframe.columns:
                self.graphics.bar_plot(self.dataframe, cat)
