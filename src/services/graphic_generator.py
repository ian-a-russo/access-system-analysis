import matplotlib.pyplot as plt
import pandas as pd
import os

class GraphicGenerator:
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def save_graphics(self, fig, filename: str):
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path)
        plt.close(fig)  

    def plot_distribution(self, dataframe: pd.DataFrame, column: str, bins: int = 30):
        series = pd.to_numeric(
            dataframe[column].astype(str).str.replace(',', '.'),
            errors='coerce'
        ).dropna()
        fig = plt.figure(figsize=(6,4))
        series.hist(bins=bins, edgecolor='black')
        plt.title(f"Distribuição de {column}")
        plt.xlabel(column)
        plt.ylabel('Frequência')
        plt.tight_layout()
        self.save_graphics(fig, f"distribution_{column}.png")

    def correlation_heatmap(self, dataframe: pd.DataFrame):
        corr = dataframe.select_dtypes(include='number').corr()
        fig = plt.figure(figsize=(8,6))
        plt.imshow(corr, interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=90)
        plt.yticks(range(len(corr)), corr.columns)
        plt.title('Heatmap de Correlação')
        plt.tight_layout()
        self.save_graphics(fig, 'heatmap.png')

    def scatter(self, dataframe: pd.DataFrame, x: str, y: str):
        fig = plt.figure(figsize=(6,4))
        plt.scatter(
            pd.to_numeric(dataframe[x].astype(str).str.replace(',', '.'), errors='coerce'),
            pd.to_numeric(dataframe[y].astype(str).str.replace(',', '.'), errors='coerce')
        )
        plt.title(f"{y} vs {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        self.save_graphics(fig, f"scatter_{x}_{y}.png")

    def line_plot(self, x, y, title: str = None, xlabel: str = None, ylabel: str = None):
        fig = plt.figure(figsize=(8,4))
        plt.plot(x, y, marker='o')
        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        fn = f"line_{xlabel or 'x'}_{ylabel or 'y'}.png"
        self.save_graphics(fig, fn)

    def bar_plot(self, df, column):
        fig = plt.figure(figsize=(6,4))
        df[column].value_counts().plot(kind='bar', edgecolor='black')
        plt.title(f'Distribuição de {column}')
        plt.xlabel(column)
        plt.ylabel('Frequência')
        plt.tight_layout()
        self.save_graphics(fig, f'bar_{column}.png')


    def time_series(self, dataframe: pd.DataFrame, date_col: str, freq: str = 'D'):
        dataframe2 = dataframe.copy()
        dataframe2[date_col] = pd.to_datetime(dataframe2[date_col], dayfirst=True, errors='coerce')
        dataframe2 = dataframe2.dropna(subset=[date_col])
        ts = dataframe2.set_index(date_col).resample(freq).size()
        self.line_plot(
            ts.index, ts.values,
            title=f'Acidentes por {freq}',
            xlabel=date_col,
            ylabel='Contagem'
        )