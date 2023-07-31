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

tipo = st.selectbox('Demonstrativo:', ['Consolidado','Individual', 'Todos'])

# ------------------------------------------------------------------------------

#df_concatenado = dfp.demonstrativos_empresa(cd_cvm)

if tipo != 'Todos':
  df_concatenado = df_concatenado[(df_concatenado['GRUPO_DFP'].str.contains(tipo))]

# ------------------------------------------------------------------------------

niveis_conta = np.sort(df_concatenado['NIVEL_CONTA'].unique())

# Definir o nível de detalhamento
nivel_conta = st.selectbox('Selecione o nivel de detalhamento:', np.sort(niveis_conta).tolist())

# ------------------------------------------------------------------------------

df_filtered = df_concatenado[df_concatenado['NIVEL_CONTA'] <= nivel_conta]

# ------------------------------------------------------------------------------

st.subheader("Demonstrativos da empresa")

# Criar um selectbox com os cmaps
selected_cmap = st.selectbox('Opções de escala de cores:', cmaps)

# Tabela com as contas da empresa
df = fun.tabela_contas_empresa(df_filtered, percentual=False)
fun.tabela_com_estilo(df, cmap=selected_cmap)

# ------------------------------------------------------------------------------

st.subheader("1. Qual a complexidade para analisar uma DFP?")

st.subheader("R: Visão Geral")

st.write(f"Número de instâncias: Há {df.shape[0]} observações e {df.shape[1]} atributos neste conjunto de dados. Destes, {len(df.select_dtypes(include=['object']).columns.tolist())} categóricos e {len(df.select_dtypes(exclude=['object']).columns.tolist())} numéricos.")

st.write(f"Os atributos categóricos são: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}")

st.write(f"Os atributos numéricos são: {', '.join(df.select_dtypes(exclude=['object']).columns.tolist())}")

data_container = st.container()
with data_container:
  metrica1, metrica2 = st.columns(2)
  with metrica1:
    st.metric("Observações (Qtd. de Contas)", df.shape[0], delta=None, delta_color="normal", help=None)
    st.metric("Categóricos", len(df.select_dtypes(include=['object']).columns.tolist()), delta=None, delta_color="normal", help=None)
  with metrica2:
    st.metric("Atributos", df.shape[1], delta=None, delta_color="normal", help=None)
    st.metric("Numéricos (Qtd. de Anos)", len(df.select_dtypes(exclude=['object']).columns.tolist()), delta=None, delta_color="normal", help=None)

# ------------------------------------------------------------------------------

st.subheader("R: Gráfico de rede - Plotly - Todas as demonstrações")

df = df_filtered.sort_values(by=['CD_CONTA_PAI','CD_CONTA']).reset_index(drop=True)
color_continuous_scale = st.selectbox('Opções de escala de cores:', px.colors.named_colorscales())
fig = fun.desenha_grafico_rede_plotly(df, node_label='CD_CONTA', title=f"Todas as demonstrações financeiras padronizadas - {denom_cia}", colorscale=color_continuous_scale)
st.plotly_chart(fig, use_container_width=True)

st.subheader("R: Gráfico de rede - NetworkX Altair - Todas as demonstrações")

# Gráfico
node_color = st.selectbox('Atributo:', ['NIVEL_CONTA:N','ST_CONTA_FIXA:N','VL_CONTA:Q','degree:Q'])
scheme_dict = {'NIVEL_CONTA:N':'categorical','ST_CONTA_FIXA:N':'cyclical','VL_CONTA:Q':'sequential_multi_hue','degree:Q':'sequential_single_hue'}
cmap = st.selectbox('Opções de escala de cores:', fun.vega_schemes(scheme_dict[node_color])) # https://vega.github.io/vega/docs/schemes/
my_chart = fun.desenha_grafico_rede(df_filtered, node_color=node_color, cmap=cmap, title=f"Todas as demonstrações financeiras padronizadas - {denom_cia}")
st.altair_chart(my_chart.interactive(), use_container_width=True) 


charts = []
dfps = df_filtered['GRUPO_DFP'].unique().tolist()
dfps.sort(key=lambda x: x.split("-")[-1].strip())
for dfp in dfps:
  df = df_filtered[(df_filtered['GRUPO_DFP'] == dfp)]
  charts.append(fun.desenha_grafico_rede(df, node_color='NIVEL_CONTA:N', cmap=cmap, title=f"{dfp} - {denom_cia}"))

concatenated_charts = []
for i in range(0, len(charts), 2):
    if i + 1 < len(charts):
        concatenated_charts.append(alt.hconcat(charts[i], charts[i + 1]))
    else:
        concatenated_charts.append(charts[i])
st.altair_chart(alt.vconcat(*concatenated_charts), use_container_width=True) 

# ------------------------------------------------------------------------------

if tipo == 'Todos':

  st.subheader("2. Qual a diferença entre analisar uma demonstração consolidada e uma individual?")
  
  st.write("As demonstrações podem ser individuais ou consolidadas. Quando forem individuais estamos falando apenas da companhia controladora.")
  st.write("No entanto, na apresentação consolidada, tanto a empresa controladora quanto suas subsidiárias são consideradas uma única entidade financeira.")
  st.write("https://www.suno.com.br/artigos/demonstracoes-financeiras")

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
    my_chart = fun.desenha_grafico_rede(df, node_color=node_color, cmap=cmap, title=f"{dfp1} - {denom_cia}")
    st.altair_chart(my_chart.interactive(), use_container_width=True) 

  with tab2:
    dfp2 = st.selectbox('DFP Individual:', list(filter(lambda k: 'Individual' in k, dfps)))
    df = df_filtered[(df_filtered['GRUPO_DFP'] == dfp2)]
    my_chart = fun.desenha_grafico_rede(df, node_color=node_color, cmap=cmap, title=f"{dfp2} - {denom_cia}")
    st.altair_chart(my_chart.interactive(), use_container_width=True) 

# ------------------------------------------------------------------------------
