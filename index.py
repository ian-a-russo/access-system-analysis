import pandas as pd
from services.LineGraphicGenerator import LineGraphicGenerator

line_graphic_generator = LineGraphicGenerator()

def generate_graphics(url):
    dados = pd.read_csv(url, sep=';', encoding='latin1', engine='python')
    
    print(dados.columns)
    
    dias = dados["dia"]
    sucesso_pct = dados["proporcao_de_acessos_bem_sucedidos"]
    erro_pct = dados["proporcao_de_operacoes_com_erros"]
    stock_pct = dados["produtos_em_estoque"]

    # Gráfico 1: Proporção de Acessos Bem-Sucedidos
    line_graphic_generator.execute(
        dias, sucesso_pct,
        'proporcao_de_acessos_bem_sucedidos',
        'Dia', 'Sucesso (%)'
    )   
    
    # Gráfico 2: Proporção de Operações com Erros
    line_graphic_generator.execute(
        dias, erro_pct,
        'proporcao_de_operacoes_com_erros',
        'Dia', 'Erro (%)'
    )   
    
    # Gráfico 3: Produtos em Estoque (%)
    line_graphic_generator.execute(
        dias, stock_pct,
        'produtos_em_estoque',
        'Dia', 'Estoque (%)'
    )

# Chamar função principal
generate_graphics('./data/acessos.csv')
