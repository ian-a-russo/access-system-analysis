import pandas as pd
from src.pipelines.accident_pipeline import AccidentPipeline

dataframe = pd.read_csv(
    './src/data/datatran2024.csv',
    sep=';',
    encoding='latin-1',
    engine='python',
)
dist_cols = [
    'pessoas',       
    'veiculos',      
    'mortos',        
    'feridos_leves', 
    'feridos_graves' 
]
pipeline = AccidentPipeline(dataframe)
pipeline.show_eda(    
    dist_cols=dist_cols,
    do_heatmap=False,
)    
pipeline.run()         
