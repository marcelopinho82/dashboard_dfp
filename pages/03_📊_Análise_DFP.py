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
import matplotlib.pyplot as plt
import plotly.express as px

# Obter a lista de todos os cmaps disponíveis
cmaps = ['']
cmaps.extend(plt.colormaps())

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Análise Exploratória DFP 📊")

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

# Definir a data de referência
datas_referencia = dfp.busca_datas_referencia(df_csv, cd_cvm)
dt_refer = st.selectbox('Selecione a data de referência:', datas_referencia)

# Definir o nível de detalhamento
nivel_conta = st.selectbox('Selecione o nivel de detalhamento:', np.sort(df_csv['NIVEL_CONTA'].unique())[::-1])

# ------------------------------------------------------------------------------

analises = []
analises.append("Dados na data de referência")
analises.append("Dados na data de referência - Gráfico de rede - NetworkX")
analises.append("Dados na data de referência - Gráfico de rede - Plotly")
analises.append("Dados na data de referência - Gráfico de barras horizontal")
analises.append("Dados na data de referência - Gráfico de barras vertical")
analises.append("Dados na data de referência - Conta X subcontas - Gráfico de barras horizontal")
analises.append("Dados na data de referência - Conta X subcontas - Gráfico de barras vertical")
analises.append("Evolução das contas")
analises.append("Evolução das contas - Gráfico de barras horizontal")
analises.append("Evolução das contas - Conta X conta")
analises.append("Evolução das contas - Conta X subcontas - Gráfico de barras horizontal")
analises.append("Evolução das contas - Conta X subcontas - Gráfico de barras vertical")

if "aberta_BPA_con" in option or "aberta_BPP_con" in option or "aberta_BP_con" in option or "aberta_BPA_ind" in option or "aberta_BPP_ind" in option or "aberta_BP_ind" in option:
  analises.append("Dados na data de referência - Treemap Plotly")
  analises.append("Dados na data de referência - Sunburst")
  analises.append("Dados na data de referência - Conta X subcontas - Treemap Squarify")
  analises.append("Dados na data de referência - Conta X subcontas - Treemap Plotly")
  analises.append("Dados na data de referência - Conta X subcontas - Sunburst")
  analises.append("Evolução das contas - Treemap Plotly")
  analises.append("Evolução das contas - Sunburst")    
  analises.append("Evolução das contas - Conta X subcontas - Treemap Squarify")    
  analises.append("Evolução das contas - Conta X subcontas - Treemap Plotly")    
  analises.append("Evolução das contas - Conta X subcontas - Sunburst")    
  analises.append("Evolução das contas - Gráfico de linhas")
  analises.append("Evolução das contas - Conta X subcontas - Gráfico de linhas")
elif "aberta_DRE_con" in option or "aberta_DRE_ind" in option or "aberta_DRA_con" in option or "aberta_DRA_ind" in option or "aberta_DVA_con" in option or "aberta_DVA_ind" in option:
  analises.append("Dados na data de referência - Waterfall")
  analises.append("Evolução das contas - Gráfico de linhas")
  analises.append("Evolução das contas - Conta X subcontas - Gráfico de linhas")
#elif "DFC_MD" in option:
#  analises.append("")
#elif "DFC_MI" in option:
#  analises.append("")
#elif "DMPL" in option:    
#  analises.append("")

analises = np.sort(analises)
analise = st.selectbox('Qual a análise gostaria de realizar?', analises)

# ------------------------------------------------------------------------------

st.subheader(analise)

# ------------------------------------------------------------------------------

if analise == "Dados na data de referência":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)

  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)
  
  # Gráfico
  st.line_chart(df, x='DS_CONTA', y=fun.retorna_colunas_data(df))  
  
  # Atributos
  fun.atributos(df.drop(['%', 'T'], axis=1), cmap=selected_cmap)
 
  # Tabs  
  tabs = st.tabs(fun.retorna_colunas_data(df))
  for i, data in enumerate(fun.retorna_colunas_data(df)):
    with tabs[i]:
      st.subheader(data)
      filtered_df = df_DFP[df_DFP['DT_FIM_EXERC'] == data].sort_values(by=['CD_CONTA_PAI', 'CD_CONTA']).reset_index(drop=True)
      # Gráfico
      pivoted_df = dfp.pivotear_tabela(filtered_df)
      st.line_chart(pivoted_df, x='DS_CONTA', y=data)
      st.bar_chart(pivoted_df, x='DS_CONTA', y=data)  
  
# ------------------------------------------------------------------------------
  
if analise == "Dados na data de referência - Gráfico de rede - NetworkX":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)

  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)
  
  # Gráfico
  node_color = st.selectbox('Atributo:', ['NIVEL_CONTA:N','ST_CONTA_FIXA:N','VL_CONTA:Q','degree:Q'])
  scheme_dict = {'NIVEL_CONTA:N':'categorical','ST_CONTA_FIXA:N':'cyclical','VL_CONTA:Q':'sequential_multi_hue','degree:Q':'sequential_single_hue'}
  cmap = st.selectbox('Opções de escala de cores:', fun.vega_schemes(scheme_dict[node_color])) # https://vega.github.io/vega/docs/schemes/
  
  # Tabs  
  tabs = st.tabs(fun.retorna_colunas_data(df))
  for i, data in enumerate(fun.retorna_colunas_data(df)):
    with tabs[i]:
      st.subheader(data)
      filtered_df = df_DFP[df_DFP['DT_FIM_EXERC'] == data].sort_values(by=['CD_CONTA_PAI', 'CD_CONTA']).reset_index(drop=True)
      # Gráfico
      my_chart = fun.desenha_grafico_rede(filtered_df, node_color=node_color, cmap=cmap, title=titulo)
      st.altair_chart(my_chart.interactive(), use_container_width=True) 

# ------------------------------------------------------------------------------

if analise == "Dados na data de referência - Gráfico de rede - Plotly":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)
  
  # Gráfico
  node_label = st.selectbox('Atributo:', ['','DS_CONTA','CD_CONTA'])  

  # Tabs  
  tabs = st.tabs(fun.retorna_colunas_data(df))
  for i, data in enumerate(fun.retorna_colunas_data(df)):
    with tabs[i]:
      st.subheader(data)
      filtered_df = df_DFP[df_DFP['DT_FIM_EXERC'] == data].sort_values(by=['CD_CONTA_PAI', 'CD_CONTA']).reset_index(drop=True)
      # Gráfico
      color_continuous_scale = st.selectbox(f'Opções de escala de cores {i+1}:', px.colors.named_colorscales())
      fig = fun.desenha_grafico_rede_plotly(filtered_df, node_label=node_label, title=titulo, colorscale=color_continuous_scale) 
      st.plotly_chart(fig, use_container_width=True)
  
# ------------------------------------------------------------------------------
  
if analise == "Dados na data de referência - Waterfall":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)
    
  # Tabs  
  tabs = st.tabs(fun.retorna_colunas_data(df))
  for i, data in enumerate(fun.retorna_colunas_data(df)):
    with tabs[i]:
      st.subheader(data)
      filtered_df = df_DFP[df_DFP['DT_FIM_EXERC'] == data].sort_values(by=['CD_CONTA_PAI', 'CD_CONTA']).reset_index(drop=True)
      # Gráfico
      fig = fun.gerar_waterfall(filtered_df, title=titulo + " - " + data)
      st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Gráfico de barras horizontal":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # Gráfico 1
  my_chart = fun.grafico_1(dfp.pivotear_tabela(df_DFP), titulo, denom_cia, dt_refer, cmap=selected_cmap)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Gráfico de barras vertical":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # Gráfico comparativo último X penúltimo exercício separado
  my_chart = fun.grafico_2(df_DFP, titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Treemap Plotly":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]
  
  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())    
  fig = fun.gerar_treemap(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)
  
# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Sunburst":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]
  
  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_sunburst(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Gráfico de barras horizontal":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  df_pai = fun.incluir_percentual(df_pai)
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0)
  #Gráfico
  my_chart = fun.grafico_1(dfp.pivotear_tabela(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)]), titulo, denom_cia, dt_refer, cmap=selected_cmap)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  df_filho = fun.incluir_percentual(df_filho)
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0) # Exibir o dataframe filtrado
  # Gráfico
  my_chart = fun.grafico_1(df_filho, titulo, denom_cia, dt_refer, cmap=selected_cmap)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Gráfico de barras vertical":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  df_pai = fun.incluir_percentual(df_pai)
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0)
  # Gráfico
  my_chart = fun.grafico_2(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  df_filho = fun.incluir_percentual(df_filho)
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0) # Exibir o dataframe filtrado
  
  # Gráfico
  my_chart = fun.grafico_2(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Treemap Squarify":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  df_pai = fun.incluir_percentual(df_pai)
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  df_filho = fun.incluir_percentual(df_filho)
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------

  # Gráfico
  atributo = st.selectbox('Atributo:', ['CD_CONTA','DS_CONTA'])
  fun.gerar_squarify(df_filho, atributo=atributo, title=titulo, cmap=selected_cmap)  

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Treemap Plotly":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  df_pai = fun.incluir_percentual(df_pai)
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  df_filho = fun.incluir_percentual(df_filho)
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------
  
  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------
  
  # Gráfico
  color = st.selectbox('Atributo:', ['VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_treemap(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------

elif analise == "Dados na data de referência - Conta X subcontas - Sunburst":

  # Busca todos os dados da empresa na data de referência selecionada
  df_DFP = dfp.dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=True)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=0)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  df_pai = fun.incluir_percentual(df_pai)
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  df_filho = fun.incluir_percentual(df_filho)
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=0) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------
  
  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------
  
  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_sunburst(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)

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
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)
  
  # Gráfico
  st.line_chart(df, x='DS_CONTA', y=fun.retorna_colunas_data(df))
  
  # Atributos
  fun.atributos(df, cmap=selected_cmap)

  # Tabs  
  tabs = st.tabs(fun.retorna_colunas_data(df))
  for i, data in enumerate(fun.retorna_colunas_data(df)):
    with tabs[i]:
      st.subheader(data)
      filtered_df = df_DFP[df_DFP['DT_FIM_EXERC'] == data].sort_values(by=['CD_CONTA_PAI', 'CD_CONTA']).reset_index(drop=True)
      # Gráfico
      pivoted_df = dfp.pivotear_tabela(filtered_df)
      
      st.write("**Gráfico de linhas**")
      st.line_chart(pivoted_df, x='DS_CONTA', y=data)
      st.write("**Gráfico de barras**")
      st.bar_chart(pivoted_df, x='DS_CONTA', y=data)
      st.write("**Gráfico de área**")
      st.area_chart(pivoted_df, x='DS_CONTA', y=data)        

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Gráfico de barras horizontal":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)

  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # Gráfico 1
  my_chart = fun.grafico_1(dfp.pivotear_tabela(df_DFP), titulo, denom_cia, dt_refer, cmap=selected_cmap)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Gráfico de linhas":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)

  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)

  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # Gráfico de linha comparativo com a evolução das contas da empresa ao longo dos anos
  my_chart = fun.grafico_3(df_DFP, titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Treemap Plotly":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]

  # Gráfico
  color = st.selectbox('Atributo:', ['VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_treemap(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Sunburst":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)
  
  # Remover contas nulas
  df_DFP = df_DFP[pd.notna(df_DFP['DS_CONTA_PAI'])]

  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_sunburst(df_DFP, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X conta":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, nivel_conta)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  contas = df_DFP['CD_CONTA'].unique()
  conta1 = st.selectbox('Conta 1', np.sort(contas).tolist()[::1])
  conta2 = st.selectbox('Conta 2', np.sort(contas).tolist()[::-1])
  
  filtered_df = df_DFP[df_DFP['CD_CONTA'].isin([conta1, conta2])].reset_index(drop=True) 
  fun.tabela_com_estilo(dfp.pivotear_tabela(filtered_df, margins=True, margins_name='Total'), cmap=selected_cmap, axis=1)
  
  # Gráfico
  st.subheader("Gráfico de barras + linhas")
  my_chart = fun.grafico_comparativo_duas_contas(df_DFP, titulo, denom_cia, dt_refer, conta1, conta2)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 
    
  # Gráfico
  df = dfp.transpor(filtered_df)
  st.subheader("Gráfico de linhas")
  st.line_chart(df)
  st.subheader("Gráfico de barras")
  st.bar_chart(df)
  st.subheader("Gráfico de área")
  st.area_chart(df)  
  
# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Gráfico de linhas":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 5)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1)
  # Gráfico
  my_chart = fun.grafico_3(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1) # Exibir o dataframe filtrado
  # Gráfico
  my_chart = fun.grafico_3(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Gráfico de barras horizontal":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1)
  # Gráfico
  my_chart = fun.grafico_1(dfp.pivotear_tabela(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)]), titulo, denom_cia, dt_refer, cmap=selected_cmap)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1) # Exibir o dataframe filtrado
  # Gráfico
  my_chart = fun.grafico_1(df_filho, titulo, denom_cia, dt_refer, cmap=selected_cmap)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Gráfico de barras vertical":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1)
  # Gráfico
  my_chart = fun.grafico_2(df_DFP[(df_DFP['CD_CONTA'] == cd_conta)], titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1) # Exibir o dataframe filtrado
  # Gráfico
  my_chart = fun.grafico_2(df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)], titulo, denom_cia, dt_refer)
  st.altair_chart(my_chart.interactive(), use_container_width=True) 

  # ----------------------------------------------------------------------------

# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Treemap Squarify":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------

  # Gráfico
  atributo = st.selectbox('Atributo:', ['CD_CONTA','DS_CONTA'])
  fun.gerar_squarify(df_filho, atributo=atributo, title=titulo, cmap=selected_cmap)  
    
# ------------------------------------------------------------------------------

elif analise == "Evolução das contas - Conta X subcontas - Treemap Plotly":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------

  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------

  # Gráfico
  color = st.selectbox('Atributo:', ['VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_treemap(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------ 

elif analise == "Evolução das contas - Conta X subcontas - Sunburst":

  # Busca todos os dados da empresa até a data de referência selecionada
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)
  
  # Criar um selectbox com os cmaps
  selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
  # Tabela com as contas da empresa
  df = fun.tabela_contas_empresa(df_DFP, percentual=False)
  fun.tabela_com_estilo(df, cmap=selected_cmap, axis=1)

  # ----------------------------------------------------------------------------

  contas_pai = df_DFP['CD_CONTA_PAI'].unique()
  cd_conta = st.selectbox('Conta', df_DFP[(df_DFP['NIVEL_CONTA'] <= nivel_conta) & (df_DFP['CD_CONTA'].isin(contas_pai))]['CD_CONTA'].unique()) # Seleciona a conta desejada

  # ----------------------------------------------------------------------------

  df = dfp.pivotear_tabela(df_DFP, index=['CD_CONTA', 'DS_CONTA', 'CD_CONTA_PAI']) # Pivotear a tabela
  df_pai = df[(df['CD_CONTA'] == cd_conta)] # Filtrar o dataframe pela conta selecionada
  fun.tabela_com_estilo(df_pai.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1)
  df_filho = df[(df['CD_CONTA_PAI'] == cd_conta)] # Filtrar o dataframe selecionando as subcontas
  st.write("Subcontas")
  fun.tabela_com_estilo(df_filho.drop(["CD_CONTA_PAI"], axis=1), cmap=selected_cmap, axis=1) # Exibir o dataframe filtrado

  # ----------------------------------------------------------------------------

  df_filho = df_DFP[(df_DFP['CD_CONTA_PAI'] == cd_conta)]

  # ----------------------------------------------------------------------------

  # Gráfico
  color = st.selectbox('Atributo:', [None, 'VL_CONTA','DS_CONTA','DS_CONTA_PAI'])
  color_continuous_scale = None
  if color == "VL_CONTA":
    color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
  fig = fun.gerar_sunburst(df_filho, color_continuous_scale=color_continuous_scale, title=titulo, color=color)
  st.plotly_chart(fig, use_container_width=True)
