%%writefile my_app/pages/02_📖_Análise_DFP.py
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image

# ------------------------------------------------------------------------------

import sys
sys.path.append('/content/marcelo/marcelo')
import marcelo as mp
import DFP as dfp
import matplotlib.pyplot as plt
import squarify
import seaborn as sns

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Análise Exploratória DFP")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

import os
DIR = "/content"
entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]
options = mp.Filter(entries, ['dfp_cia_aberta'])
options.remove("dfp_cia_aberta.csv")
option = st.selectbox('Qual o conjunto de dados gostaria de analisar?', np.sort(options))
df_csv = pd.read_csv(option)
df_csv.drop_duplicates(inplace=True)

# ------------------------------------------------------------------------------

titulo = dfp.retorna_titulo(option)

st.write(f'Você escolheu: {option} ({titulo})')

# ------------------------------------------------------------------------------

# Definir a empresa a ser analisada
denom_cia = st.selectbox('Qual a empresa gostaria de analisar?', df_csv['DENOM_CIA'].unique())
cd_cvm = dfp.busca_cod_cvm_empresa(denom_cia)

# ------------------------------------------------------------------------------

analises = [
  "Dados na data de referência",
  "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras horizontal",
  "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras vertical",
  "Dados na data de referência - Conta X subcontas - Gráfico de barras horizontal",
  "Dados na data de referência - Conta X subcontas - Gráfico de barras vertical",
  "Dados na data de referência - Conta X subcontas - Treemap",
  "Evolução das contas",
  "Evolução das contas - Gráfico de barras horizontal",
  "Evolução das contas - Gráfico de linhas",
  "Evolução das contas - Conta X conta",
  "Evolução das contas - Conta X subcontas - Gráfico de linhas",
  "Evolução das contas - Conta X subcontas - Gráfico de barras horizontal",
  "Evolução das contas - Conta X subcontas - Gráfico de barras vertical",
  "Evolução das contas - Conta X subcontas - Treemap"
]

analise = st.selectbox('Qual a análise gostaria de realizar?', analises)

# ------------------------------------------------------------------------------

# Definir a data de referência
datas_referencia = dfp.busca_datas_referencia(df_csv, cd_cvm)
dt_refer = st.selectbox('Selecione a data de referência:', datas_referencia)

# Definir o nível de detalhamento
nivel_conta = st.selectbox('Selecione o nivel de detalhamento:', np.sort(df_csv['NIVEL_CONTA'].unique()))

# ------------------------------------------------------------------------------

st.subheader(analise)

# ------------------------------------------------------------------------------

if analise == "Dados na data de referência":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)

  st.write("Tabela com as contas da empresa na data de referência")
  st.dataframe(df_DFP)

  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras horizontal":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Gráfico 1
  dfp.grafico_1(dfp.pivotear_tabela(df_DFP), titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras vertical":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Gráfico comparativo último X penúltimo exercício separado
  dfp.grafico_2(df_DFP, titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Gráfico de barras horizontal":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  dfp.grafico_1(dfp.pivotear_tabela(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)]), titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  dfp.grafico_1(df_filho, titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Gráfico de barras vertical":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  dfp.grafico_2(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  dfp.grafico_2(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Treemap":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------

  cores = dfp.retorna_cores(len(df_filho))
  for data in dfp.retorna_colunas_data(df_filho):
    fig = plt.figure()
    sns.set_style(style="whitegrid") # Set seaborn plot style
    sizes = df_filho[data].values # Proporção das categorias
    label=df_filho["CD_CONTA"]
    squarify.plot(sizes=sizes, label=label, alpha=0.6, color=cores).set(title=f'{data}')
    plt.axis('off')
    plt.show()
    st.pyplot(fig)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Tabela pivoteada com a evolução das contas da empresa nas datas de referência
  st.dataframe(dfp.pivotear_tabela(df_DFP))

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Gráfico de barras horizontal":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Tabela legenda
  st.dataframe(dfp.pivotear_tabela(df_DFP))

  # Gráfico 1
  dfp.grafico_1(dfp.pivotear_tabela(df_DFP), titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Gráfico de linhas":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Tabela legenda
  st.dataframe(dfp.pivotear_tabela(df_DFP))

  # Gráfico de linha comparativo com a evolução das contas da empresa ao longo dos anos
  dfp.grafico_3(df_DFP, titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X conta":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Gráfico de linha comparativo com a evolução das contas da empresa ao longo dos anos
  conta1 = st.selectbox('Conta 1', df_DFP['CD_CONTA'].unique())
  conta2 = st.selectbox('Conta 2', df_DFP['CD_CONTA'].unique())
  dfp.grafico_comparativo_duas_contas(df_DFP, titulo, denom_cia, dt_refer, conta1, conta2)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Gráfico de linhas":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  dfp.grafico_3(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  dfp.grafico_3(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------


# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Gráfico de barras horizontal":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  dfp.grafico_1(dfp.pivotear_tabela(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)]), titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  dfp.grafico_1(df_filho, titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Gráfico de barras vertical":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  dfp.grafico_2(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  dfp.grafico_2(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Treemap":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)

  # ----------------------------------------------------------------------------

  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] == nivel_conta)]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  st.dataframe(df_pai)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------

  cores = dfp.retorna_cores(len(df_filho))
  for data in dfp.retorna_colunas_data(df_filho):
    fig = plt.figure()
    sns.set_style(style="whitegrid") # Set seaborn plot style
    sizes = df_filho[data].values # Proporção das categorias
    label=df_filho["CD_CONTA"]
    squarify.plot(sizes=sizes, label=label, alpha=0.6, color=cores).set(title=f'{data}')
    plt.axis('off')
    plt.show()
    st.pyplot(fig)

# ------------------------------------------------------------------------------
