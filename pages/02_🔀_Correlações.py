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

st.markdown("# Correlações 🔀")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

import os
DIR = "./"
entries = [entry for entry in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, entry))]



options = mp.Filter(entries, ['dfp_cia_aberta_BP_con', 'dfp_cia_aberta_BP_ind','dfp_cia_aberta_BPA_con', 'dfp_cia_aberta_BPA_ind','dfp_cia_aberta_BPP_con', 'dfp_cia_aberta_BPP_ind'])
option_arquivo1 = st.selectbox('Selecione o primeiro conjunto de dados', np.sort(options))
df_csv1 = pd.read_csv(option_arquivo1)
df_csv1.drop_duplicates(inplace=True)
titulo1 = dfp.retorna_titulo(option_arquivo1)
st.write(f'Você escolheu: {option_arquivo1} ({titulo1})')

options.remove(option_arquivo1)

options = mp.Filter(entries, ['dfp_cia_aberta_DRE_con', 'dfp_cia_aberta_DRE_ind'])
option_arquivo2 = st.selectbox('Selecione o segundo conjunto de dados', np.sort(options))
df_csv2 = pd.read_csv(option_arquivo2)
df_csv2.drop_duplicates(inplace=True)
titulo2 = dfp.retorna_titulo(option_arquivo2)
st.write(f'Você escolheu: {option_arquivo2} ({titulo2})')

# ------------------------------------------------------------------------------

# Definir a empresa a ser analisada
# https://discuss.streamlit.io/t/multiselect-widget-data-displayed-in-alphabetical-order/21617/2
denom_cia = st.selectbox('Qual a empresa gostaria de analisar?', dfp.lista_empresas())
cd_cvm = dfp.busca_cod_cvm_empresa(denom_cia)

# ------------------------------------------------------------------------------

# Definir a data de referência
datas_referencia = dfp.busca_datas_referencia(df_csv1, cd_cvm)
dt_refer = st.selectbox('Selecione a data de referência:', datas_referencia)

# Definir o nível de detalhamento
nivel_conta = st.selectbox('Selecione o nivel de detalhamento:', np.sort(df_csv1['NIVEL_CONTA'].unique())[::-1])

# ------------------------------------------------------------------------------

# Busca todos os dados da empresa até a data de referência selecionada
df_DFP1 = dfp.dados_da_empresa(df_csv1, cd_cvm, dt_refer, nivel_conta)
df_DFP2 = dfp.dados_da_empresa(df_csv2, cd_cvm, dt_refer, nivel_conta)

df_DFP = pd.concat([df_DFP1, df_DFP2])

# ------------------------------------------------------------------------------
  
# Criar um selectbox com os cmaps
selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)
  
# Tabela com as contas da empresa
df = fun.tabela_contas_empresa(df_DFP, percentual=False)
fun.tabela_com_estilo(df, cmap=selected_cmap)

# ------------------------------------------------------------------------------

# Dados

df = fun.bruxaria(df)

# ------------------------------------------------------------------------------

st.write("Tabela transposta com os dados da empresa")
st.write(df)

# ------------------------------------------------------------------------------

st.subheader("Tabela de correlações identificadas")
dfCorr = mp.top_entries(df)
dfCorr.dropna(axis=0, inplace=True)
st.write(dfCorr)

# ------------------------------------------------------------------------------

st.subheader("Tabela de correlações filtrada")

dfCorr = mp.top_entries(df)
dfCorr.dropna(axis=0, inplace=True)
dfCorr = dfCorr.loc[((dfCorr['Correlação'] >= .5) | (dfCorr['Correlação'] <= -.5)) & (dfCorr['Correlação'] !=1.000)]
st.write(dfCorr)

# ------------------------------------------------------------------------------

st.subheader("Correlação de variáveis independentes com a variável dependente")

options = dfCorr.sort_values(by="Atributo X")["Atributo X"].unique().tolist()
options.extend(dfCorr.sort_values(by="Atributo Y")["Atributo Y"].unique().tolist())
target = st.selectbox('Selecione a coluna alvo (target):', options[::-1])

# ------------------------------------------------------------------------------

# https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

# Criar um controle deslizante para a variação da correlação
min_corr = 0.5
max_corr = 1.0
selected_corr = st.slider('Selecione a variação da correlação', min_corr, max_corr, (min_corr, max_corr))
selected_corr_min = selected_corr[0]
selected_corr_max = selected_corr[1]

dfCorr = df.corr()
filteredDf = dfCorr[[target]]
filteredDf = filteredDf.loc[(((filteredDf[target] >= selected_corr_min) & (filteredDf[target] <= selected_corr_max)) | ((filteredDf[target] >= selected_corr_max*-1) & (filteredDf[target] <= selected_corr_min*-1))) & (filteredDf[target] !=1.000)]
filteredDf = filteredDf.dropna(axis=0).sort_values(by=target, ascending=False)
st.write(filteredDf)

# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(8, 12))
if selected_cmap == '':
  heatmap = sns.heatmap(filteredDf, vmin=-1, vmax=1, annot=True, cmap="coolwarm")
else:
  heatmap = sns.heatmap(filteredDf, vmin=-1, vmax=1, annot=True, cmap=selected_cmap)
heatmap.set_title(f'Atributos correlacionados com {target}', fontdict={'fontsize':18}, pad=16);
st.pyplot(fig)

# ------------------------------------------------------------------------------

options = filteredDf[[target]].dropna(axis=0).sort_values(by=target, ascending=False)
options = options.index.tolist()

st.subheader("Gráfico de linhas")
y_column = st.selectbox('Selecione a coluna do eixo y:', options)
fig = plt.figure(figsize=(10,5))
sns.lineplot(data=df, x=target, y=y_column, color='r')
st.pyplot(fig)

# ------------------------------------------------------------------------------
