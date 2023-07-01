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
import marcelo as mp
import DFP as dfp
import funcoes as fun

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Análise Exploratória DFP")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

import os
DIR = "./"
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
# https://discuss.streamlit.io/t/multiselect-widget-data-displayed-in-alphabetical-order/21617/2
denom_cia = st.selectbox('Qual a empresa gostaria de analisar?', df_csv.sort_values(by="DENOM_CIA").DENOM_CIA.unique())
cd_cvm = dfp.busca_cod_cvm_empresa(denom_cia)

# ------------------------------------------------------------------------------

analises = [
  "Dados na data de referência",
  "Dados na data de referência - Gráfico de rede - NetworkX",
  "Dados na data de referência - Gráfico de rede - Plotly",
  "Dados na data de referência - Waterfall",
  "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras horizontal",
  "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras vertical",
  "Dados na data de referência - Último X Penúltimo exercício - Treemap Plotly",
  "Dados na data de referência - Último X Penúltimo exercício - Sunburst",
  "Dados na data de referência - Conta X subcontas - Gráfico de barras horizontal",
  "Dados na data de referência - Conta X subcontas - Gráfico de barras vertical",
  "Dados na data de referência - Conta X subcontas - Treemap Squarify",
  "Dados na data de referência - Conta X subcontas - Treemap Plotly",
  "Dados na data de referência - Conta X subcontas - Sunburst",
  "Evolução das contas",
  "Evolução das contas - Gráfico de barras horizontal",
  "Evolução das contas - Gráfico de linhas",
  "Evolução das contas - Treemap Plotly",
  "Evolução das contas - Sunburst",
  "Evolução das contas - Conta X conta",
  "Evolução das contas - Conta X subcontas - Gráfico de linhas",
  "Evolução das contas - Conta X subcontas - Gráfico de barras horizontal",
  "Evolução das contas - Conta X subcontas - Gráfico de barras vertical",
  "Evolução das contas - Conta X subcontas - Treemap Squarify",
  "Evolução das contas - Conta X subcontas - Treemap Plotly",
  "Evolução das contas - Conta X subcontas - Sunburst"
  
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

  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))
  
# ------------------------------------------------------------------------------
  
if analise == "Dados na data de referência - Gráfico de rede - NetworkX":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Selecionar apenas o último exercício
  df_DFP = df_DFP[(df_DFP['ORDEM_EXERC'] == "ÚLTIMO")]
  df = df_DFP.sort_values(by=['CD_CONTA_PAI','CD_CONTA']).reset_index(drop=True)

  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))  
  
  # Gráfico
  node_color = st.selectbox('Atributo:', ['NIVEL_CONTA:N','ST_CONTA_FIXA:N','VL_CONTA:Q','degree:Q'])
  cmap = st.selectbox('Mapas de cores:', ['viridis','set1','yellowgreen','blues'])
  fun.desenha_grafico_rede(df, node_color=node_color, cmap=cmap, title=titulo)

# ------------------------------------------------------------------------------

if analise == "Dados na data de referência - Gráfico de rede - Plotly":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Selecionar apenas o último exercício
  df_DFP = df_DFP[(df_DFP['ORDEM_EXERC'] == "ÚLTIMO")]
  df = df_DFP.sort_values(by=['CD_CONTA_PAI','CD_CONTA']).reset_index(drop=True)

  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))  
  
  # Gráfico
  node_label = st.selectbox('Atributo:', ['','DS_CONTA','CD_CONTA'])
  fun.desenha_grafico_rede_plotly(df, node_label=node_label, title=titulo)
  
# ------------------------------------------------------------------------------
  
if analise == "Dados na data de referência - Waterfall":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Selecionar apenas o último exercício
  df_DFP = df_DFP[(df_DFP['ORDEM_EXERC'] == "ÚLTIMO")]
  df = df_DFP.sort_values(by=['CD_CONTA_PAI','CD_CONTA']).reset_index(drop=True)

  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))  
  
  # Gráfico
  fun.gerar_waterfall(df, title=titulo)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras horizontal":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))

  # Gráfico 1
  fun.grafico_1(dfp.pivotear_tabela(df_DFP), titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Último X Penúltimo exercício - Gráfico de barras vertical":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))

  # Gráfico comparativo último X penúltimo exercício separado
  fun.grafico_2(df_DFP, titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Último X Penúltimo exercício - Treemap Plotly":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]
  
  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_treemap(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  
# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Último X Penúltimo exercício - Sunburst":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]
  
  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_sunburst(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
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
  fun.grafico_1(dfp.pivotear_tabela(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)]), titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  fun.grafico_1(df_filho, titulo, denom_cia, dt_refer)

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
  fun.grafico_2(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  fun.grafico_2(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Treemap Squarify":

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

  # Gráfico
  atributo = st.selectbox('Atributo:', ['CD_CONTA','DS_CONTA'])
  fun.gerar_squarify(df_filho, atributo=atributo, title=titulo)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Treemap Plotly":

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
  
  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------
  
  # Gráfico
  color = st.selectbox('Atributo:', ['VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_treemap(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Sunburst":

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
  
  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------
  
  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_sunburst(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
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
  fun.grafico_1(dfp.pivotear_tabela(df_DFP), titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Gráfico de linhas":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Tabela legenda
  st.dataframe(dfp.pivotear_tabela(df_DFP))

  # Gráfico de linha comparativo com a evolução das contas da empresa ao longo dos anos
  fun.grafico_3(df_DFP, titulo, denom_cia, dt_refer)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Treemap Plotly":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]

  # Gráfico
  color = st.selectbox('Atributo:', ['VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_treemap(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Sunburst":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]

  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_sunburst(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X conta":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  st.dataframe(dfp.pivotear_tabela(df_DFP))

  # Gráfico de linha comparativo com a evolução das contas da empresa ao longo dos anos
  conta1 = st.selectbox('Conta 1', df_DFP['CD_CONTA'].unique())
  conta2 = st.selectbox('Conta 2', df_DFP['CD_CONTA'].unique())
  fun.grafico_comparativo_duas_contas(df_DFP, titulo, denom_cia, dt_refer, conta1, conta2)

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
  fun.grafico_3(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  fun.grafico_3(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)

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
  fun.grafico_1(dfp.pivotear_tabela(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)]), titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  fun.grafico_1(df_filho, titulo, denom_cia, dt_refer)

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
  fun.grafico_2(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  st.dataframe(df_filho) # Exibir o dataframe filtrado
  fun.grafico_2(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Treemap Squarify":

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

  # Gráfico
  atributo = st.selectbox('Atributo:', ['CD_CONTA','DS_CONTA'])
  fun.gerar_squarify(df_filho, atributo=atributo, title=titulo)
    
# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Treemap Plotly":

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

  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------

  # Gráfico
  color = st.selectbox('Atributo:', ['VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_treemap(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)

# ------------------------------------------------------------------------------ 

elif analise == "Evolução das contas - Conta X subcontas - Sunburst":

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

  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------

  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', ['Viridis','Greys','YlGnBu','Greens','YlOrRd','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','Hot','Blackbody','Earth','Electric'])
  fun.gerar_sunburst(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
