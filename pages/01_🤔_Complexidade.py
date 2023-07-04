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

st.markdown("# Complexidade 🤔")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

# Definir a empresa a ser analisada
# https://discuss.streamlit.io/t/multiselect-widget-data-displayed-in-alphabetical-order/21617/2
denom_cia = st.selectbox('Qual a empresa gostaria de analisar?', dfp.lista_empresas())
cd_cvm = dfp.busca_cod_cvm_empresa(denom_cia)

# ------------------------------------------------------------------------------

analise = st.selectbox('Análise:', ['Último balanço enviado a CVM', 'Histórico'])
if analise == "Último balanço enviado a CVM":
  df_concatenado = dfp.demonstrativos_empresa_na_data_referencia(cd_cvm)
else:
  df_concatenado = dfp.demonstrativos_empresa(cd_cvm)

# ------------------------------------------------------------------------------

niveis_conta = np.sort(df_concatenado['NIVEL_CONTA'].unique())

st.subheader("Dados brutos")
st.write(df_concatenado)

# ------------------------------------------------------------------------------

# Definir o nível de detalhamento
nivel_conta = st.selectbox('Selecione o nivel de detalhamento:', np.sort(niveis_conta).tolist()[::-1])

# ------------------------------------------------------------------------------

df_filtered = df_concatenado[df_concatenado['NIVEL_CONTA'] <= nivel_conta]

# ------------------------------------------------------------------------------

st.subheader("1. Qual a complexidade para analisar uma DFP?")

st.subheader("R: Gráfico de rede - Plotly - Todas as demonstrações")

df = df_concatenado[(df_concatenado['NIVEL_CONTA'] <= nivel_conta)]
df = df.sort_values(by=['CD_CONTA_PAI','CD_CONTA']).reset_index(drop=True)
fun.desenha_grafico_rede_plotly(df, node_label='CD_CONTA', title=f"Todas as demonstrações financeiras padronizadas - {denom_cia}")

st.subheader("R: Gráfico de rede - NetworkX Altair - Todas as demonstrações")

# Gráfico
node_color = st.selectbox('Atributo:', ['NIVEL_CONTA:N','ST_CONTA_FIXA:N','VL_CONTA:Q','degree:Q'])
cmap = st.selectbox('Mapas de cores:', ['viridis','set1','yellowgreen','blues']) # https://vega.github.io/vega/docs/schemes/

df = df_concatenado[(df_concatenado['NIVEL_CONTA'] <= nivel_conta)]
fun.desenha_grafico_rede(df, node_color=node_color, cmap=cmap, title=f"Todas as demonstrações financeiras padronizadas - {denom_cia}")

# ------------------------------------------------------------------------------

st.subheader("2. Qual a diferença entre analisar uma demonstração consolidada e uma individual?")

st.subheader("R: Métricas com o quantitativo de contas de cada demonstração")
resultado = df_filtered['GRUPO_DFP'].value_counts().reset_index()
resultado.columns = ['GRUPO_DFP', 'Total']
resultado['DFP'] = resultado['GRUPO_DFP'].str.split(' - ').str[1]
resultado['Tipo'] = resultado['GRUPO_DFP'].str.split(' - ').str[0]
resultado = resultado[['DFP', 'Tipo', 'Total']]
resultado['Delta'] = resultado.groupby('DFP')['Total'].diff()
resultado = resultado.sort_values(by=['DFP'])
df = resultado
col1, col2 = st.columns(2)
col1.write("**Consolidado**")
col2.write("**Individual**")
df = resultado[resultado['Tipo'].str.contains("Consolidado")].sort_values(by=['DFP'])
for index, row in df.iterrows():
    grupo_dfp = row['DFP']
    quantitativo = row['Total']
    delta = row['Delta']# if pd.notnull(row['Delta']) else None
    delta_color = "normal" if pd.notnull(delta) else "off"
    col1.metric(label=f"{grupo_dfp}", value=quantitativo, delta=delta, delta_color=delta_color)

df = resultado[resultado['Tipo'].str.contains("Individual")].sort_values(by=['DFP'])
for index, row in df.iterrows():
    grupo_dfp = row['DFP']
    quantitativo = row['Total']
    delta = row['Delta']# if pd.notnull(row['Delta']) else None
    delta_color = "normal" if pd.notnull(delta) else "off"
    col2.metric(label=f"{grupo_dfp}", value=quantitativo, delta=delta, delta_color=delta_color)

st.subheader("R: Gráfico de rede - NetworkX Altair - Demonstrações Consolidadas / Individuais")

dfps = df_filtered.sort_values(by="GRUPO_DFP").GRUPO_DFP.unique().tolist()

# Tabs
tab1, tab2 = st.tabs(["Consolidado", "Individual"])  
with tab1:
  dfp1 = st.selectbox('DFP Consolidado:', list(filter(lambda k: 'Consolidado' in k, dfps)))
  df = df_filtered[(df_filtered['GRUPO_DFP'] == dfp1)]
  fun.desenha_grafico_rede(df, node_color=node_color, cmap=cmap, title=f"{dfp1} - {denom_cia}")
  st.write(df.describe())
with tab2:
  dfp2 = st.selectbox('DFP Individual:', list(filter(lambda k: 'Individual' in k, dfps)))
  df = df_filtered[(df_filtered['GRUPO_DFP'] == dfp2)]
  fun.desenha_grafico_rede(df, node_color=node_color, cmap=cmap, title=f"{dfp2} - {denom_cia}")
  st.write(df.describe())

# ------------------------------------------------------------------------------
