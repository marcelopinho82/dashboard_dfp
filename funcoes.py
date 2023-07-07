import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import squarify
import seaborn as sns
import DFP as dfp
import marcelo as mp

# ------------------------------------------------------------------------------

def incluir_percentual(df):
  colunas = retorna_colunas_data(df)
  df['%'] = ((df[colunas[1]] - df[colunas[0]]) / df[colunas[0]]) * 100
  df['T'] = np.select([df['%'].lt(0), df['%'].gt(0), df['%'].eq(0)], [u"\u2193", u"\u2191", u"\u2192"]) # https://stackoverflow.com/questions/59926609/how-can-add-trend-arrow-in-python-pandas  
  return df

def tabela_contas_empresa(df_DFP, percentual=True):
  st.write("Tabela pivoteada com as contas da empresa na data de referência")
  df = dfp.pivotear_tabela(df_DFP)
  if percentual:  
    df = incluir_percentual(df)
  st.write(mp.filter_dataframe(df))
  return df

# ------------------------------------------------------------------------------

def filtrar_df(df, criterios):
    filtered_df = df.copy()    
    # Aplicar os critérios de filtro
    for column, value in criterios.items():
        filtered_df = filtered_df[(filtered_df[column].str.contains(value))]    
    return filtered_df
    
def retorna_linhas(df_concatenado, criterios1, criterios2):
  df1 = retorna_linha(df_concatenado, criterios1)
  df2 = retorna_linha(df_concatenado, criterios2)
  df = pd.concat([df1, df2], ignore_index=True)
  st.write(df)
  return df

def retorna_linha(df_concatenado, criterios):
  df = filtrar_df(df_concatenado, criterios).sort_values(by=['DT_FIM_EXERC'])
  df = dfp.pivotear_tabela(df).max().to_frame().T
  return df
  
def calcula_indicador(df, nome):
  indicador = (df.iloc[0][retorna_colunas_data(df)] / df.iloc[1][retorna_colunas_data(df)]) * 100
  df_resultado = pd.DataFrame(indicador)
  df_resultado.columns = [nome]
  return df_resultado
  
# ------------------------------------------------------------------------------

def criar_paleta_cores(qtd_cores, esquema_cor):
    cmap = getattr(cm, esquema_cor)
    cores = [cmap(x) for x in range(qtd_cores)]
    #cores_hex = [cm.colors.rgb2hex(cor[:3]) for cor in cores]
    cores_hex = [cm.colors.rgb2hex(cmap(i)[:3]) for i in np.linspace(0, 1, qtd_cores)]
    return cores_hex
    
# ------------------------------------------------------------------------------

def retorna_cores(x):
  # https://colordesigner.io/gradient-generator
  # cores = ['#fafa6e','#d7f171','#b5e877','#95dd7d','#77d183','#5bc489','#3fb78d','#23aa8f','#009c8f','#008d8c','#007f86','#0b717e','#1c6373','#255566','#2a4858']  
  cores = criar_paleta_cores(x, "viridis")
  return cores[0:x]
    
# ------------------------------------------------------------------------------

def retorna_colunas_data(df):

  # https://stackoverflow.com/questions/3640359/regular-expressions-search-in-list
  import re
  mylist = df.columns.to_list()
  r = re.compile(r'\d{1,4}-\d{1,2}-\d{1,2}')
  newlist = list(filter(r.match, mylist)) # Read Note below

  return newlist

# ------------------------------------------------------------------------------

def gerar_squarify(df, atributo="CD_CONTA", title=None):
  cores = retorna_cores(len(df))
  for data in retorna_colunas_data(df):
    fig = plt.figure()
    sns.set_style(style="whitegrid") # Set seaborn plot style
    
    # https://stackoverflow.com/questions/69566317/python-float-division-by-zero-when-plotting-treemap-chart
    df2 = df.loc[df[data] != 0]
    
    sizes = df2[data].values # Proporção das categorias
    label=df2[atributo]
    squarify.plot(sizes=sizes, label=label, alpha=0.6, color=cores).set(title=title + "\n" + f"Data de Referência: {data}")
    plt.axis('off')
    plt.show()
    st.pyplot(fig)

# ------------------------------------------------------------------------------

# https://stackoverflow.com/questions/72181211/grouped-bar-charts-in-altair-using-two-different-columns

# ------------------------------------------------------------------------------

def graficos_analise_inicial(df):
  # Verificar a distribuição das colunas numéricas
  print("Distribuição das colunas numéricas:")
  num_cols = df.select_dtypes(include='number').columns.to_list()
  for col in num_cols:
    print(col)
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna())
    plt.title(f"Distribuição de {col}")
    plt.show()

  # Verificar a contagem das colunas categóricas
  print("Contagem das colunas categóricas:")
  cat_cols = df.select_dtypes(include='object').columns.to_list()
  for col in cat_cols:
    print(col)
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Contagem de {col}")
    plt.xticks(rotation=90)
    plt.show()

# ------------------------------------------------------------------------------

def grafico_1(df_ref, titulo, denom_cia, dt_refer):

  df = df_ref
  
  #contas_pai = list(df['CD_CONTA_PAI'].unique())
  #contas_pai.sort(reverse=True)
  #filtro_contas_pai = alt.selection_single(
  #    name='Contas',
  #    fields=['CD_CONTA_PAI'],
  #    bind=alt.binding_select(options=contas_pai),
  #    init={'CD_CONTA_PAI': contas_pai[0]}
  #)

  my_chart = alt.Chart(df).mark_bar().encode(
      x='Valor (R$):Q',
      y=alt.Y('ORDEM_EXERC:N', axis=alt.Axis(title='')),
      #color=alt.Color('ORDEM_EXERC:N', legend=alt.Legend(title='Exercício'), scale=alt.Scale(domain=retorna_colunas_data(df), range=retorna_cores(len(retorna_colunas_data(df))))),
      #color=alt.Color('Valor (R$):Q'),
      color=alt.Color('Valor (R$):Q', legend=alt.Legend(title='Valor (R$)'), scale=alt.Scale(range=retorna_cores(len(retorna_colunas_data(df))))),
      row=alt.Column('CD_CONTA:N', header=alt.Header(labelAngle=0), title="Conta"),
      tooltip=['CD_CONTA','DS_CONTA','ORDEM_EXERC:O','Valor (R$):Q']
  ).transform_fold(
      as_=['ORDEM_EXERC', 'Valor (R$)'],
      fold=retorna_colunas_data(df)
  ).properties(
      #width=800,
      #height=400,
      title={
          "text": f"{titulo} - {denom_cia}",
          "subtitle": f"Data de Referência: {dt_refer}",
          "fontSize": 16,
          "fontWeight": "bold"
      }
  )#.add_selection(filtro_contas_pai).transform_filter(filtro_contas_pai)

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)  

# ------------------------------------------------------------------------------

def grafico_2(df, titulo, denom_cia, dt_refer):

  #datas = list(df['DT_REFER'].unique())
  #datas.sort(reverse=True)

  #filtro_datas = alt.selection_single(
  #    name='Datas',
  #    fields=['DT_REFER'],
  #    bind=alt.binding_select(options=datas),
  #    init={'DT_REFER': datas[0]}
  #)

  #filtro_conta_fixa = alt.selection_single(
  #    name='Conta Fixa',
  #    fields=['ST_CONTA_FIXA'],
  #    bind=alt.binding_select(options=["S","N"]),
  #    init={'ST_CONTA_FIXA': "S"}
  #)

  my_chart = alt.Chart(df).mark_bar().encode(
      y=alt.Y('DS_CONTA:N', axis=alt.Axis(title='Conta', labelAngle=-45), sort=alt.EncodingSortField("CD_CONTA", order="ascending")),
      x=alt.X('VL_CONTA:Q', axis=alt.Axis(title='Valor (R$)', format='$.2f')),
      color=alt.condition(
          alt.datum.VL_CONTA > 0,
          alt.value('#85bb65'), # Cor das barras quando o valor é positivo
          alt.value('#FF4136')  # Cor das barras quando o valor é negativo
      ),
      tooltip=df.columns.tolist()
  ).properties(
      width=800,
      height=200,
      title={
          "text": f"{titulo} - {denom_cia}",
          "fontSize": 16,
          "fontWeight": "bold"
      }
  ).facet(
      row=alt.Column('ORDEM_EXERC:N', header=alt.Header(labelAngle=-90), title="Ordem Exercício")
  ).resolve_scale(
      x='independent'
  )#.add_selection(filtro_datas).transform_filter(filtro_datas).add_selection(filtro_conta_fixa).transform_filter(filtro_conta_fixa)

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)  

# ------------------------------------------------------------------------------

def grafico_3(df, titulo, denom_cia, dt_refer):

  grafico_linha1 = alt.Chart(df[df['ORDEM_EXERC'] == "ÚLTIMO"]).encode(
      x=alt.X('DT_REFER:T', axis=alt.Axis(title='Ano')),
      y=alt.Y('VL_CONTA:Q', axis=alt.Axis(title='Valor (R$)', format='$.2f'), sort=alt.EncodingSortField("CD_CONTA")),
      color=alt.Color('DS_CONTA:N', legend=alt.Legend(title="Conta")),
      tooltip=['CD_CONTA','DS_CONTA','ORDEM_EXERC','DT_FIM_EXERC','VL_CONTA']
  )

  my_chart = alt.layer(
    grafico_linha1.mark_line(), grafico_linha1.mark_square()
  ).properties(
      width=800,
      height=400,
      title={
          "text": f"{titulo} - {denom_cia}",
          "fontSize": 16,
          "fontWeight": "bold"
      }
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)  

# ------------------------------------------------------------------------------ 

def grafico_comparativo_duas_contas(df_ref, titulo, denom_cia, dt_refer, cd_conta_1, cd_conta_2):

  df1 = df_ref[(df_ref['CD_CONTA'] == cd_conta_1)]
  df2 = df_ref[(df_ref['CD_CONTA'] == cd_conta_2)]
  df = pd.concat([df1, df2])

  grafico_barras1 = alt.Chart(df1).encode(
      x=alt.X('DT_REFER:N', axis=alt.Axis(title='Data de Referência')),
      y=alt.Y('VL_CONTA:Q', axis=alt.Axis(title='Valor (R$)', format='$.2f')),
      color=alt.condition(
          alt.datum.VL_CONTA > 0,
          alt.value('#85bb65'), # Cor das barras quando o valor é positivo
          alt.value('#FF4136')  # Cor das barras quando o valor é negativo
      ),
      tooltip=df_ref.columns.tolist()
  )

  grafico_barras2 = alt.Chart(df2).encode(
      x=alt.X('DT_REFER:N', axis=alt.Axis(title='Data de Referência')),
      y=alt.Y('VL_CONTA:Q', axis=alt.Axis(title='Valor (R$)', format='$.2f')),
      color=alt.condition(
          alt.datum.VL_CONTA > 0,
          alt.value('#85bb65'), # Cor das barras quando o valor é positivo
          alt.value('#FF4136')  # Cor das barras quando o valor é negativo
      ),
      tooltip=df_ref.columns.tolist()
  )

  grafico_linha1 = alt.Chart(df_ref[df_ref['ORDEM_EXERC'] == "ÚLTIMO"]).encode(
      x=alt.X('DT_REFER:N', axis=alt.Axis(title='Ano')),
      y=alt.Y('VL_CONTA:Q', axis=alt.Axis(title='Valor (R$)', format='$.2f'), sort=alt.EncodingSortField("CD_CONTA")),
      color=alt.Color('DS_CONTA:N', legend=alt.Legend(title="Conta")),
      tooltip=df_ref.columns.tolist()
  )

  my_chart = alt.layer(
    grafico_barras1.mark_bar(), grafico_barras1.mark_line(), grafico_barras1.mark_square(), grafico_barras2.mark_bar(), grafico_barras2.mark_line(), grafico_barras2.mark_square(),

  ).properties(
      width=800,
      height=400,
      title={
          "text": f"{titulo} - {denom_cia}",
          "subtitle": f"Data de Referência: {dt_refer} - Comparativo Conta {cd_conta_1} X {cd_conta_2}",
          "fontSize": 16,
          "fontWeight": "bold"
      }
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)  

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

# Funções Sankey

import plotly.graph_objects as go

# ------------------------------------------------------------------------------ 

# ------------------------------------------------------------------------------ 

# ------------------------------------------------------------------------------ 

def create_source_target_value(df, cat_cols, value_cols):

    # transform df into a source-target-value pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','value']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','value']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'value':'sum'}).reset_index()


    return sourceTargetDf

# ------------------------------------------------------------------------------ 
    
# https://www.tutorialspoint.com/python-plotly-how-to-define-the-structure-of-a-sankey-diagram-using-a-pandas-dataframe

import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------------------ 
# ------------------------------------------------------------------------------ 
# ------------------------------------------------------------------------------ 
# ------------------------------------------------------------------------------ 
# ------------------------------------------------------------------------------ 

def get_ds_conta(df, cd_conta):
    ds_conta = df.loc[df['CD_CONTA'] == cd_conta, 'DS_CONTA']
    if not ds_conta.empty:
        return ds_conta.values[0]
    else:
        return ""

def find_index_by_cd_conta(df, cd_conta):
    index = df.index[df['CD_CONTA'] == cd_conta]
    if len(index) > 0:
        return index[0]
    else:
        return None

# ----------------------------------------------------------------------------------------------------

# Funções TreeMap

import plotly.express as px

def gerar_treemap(df, path=['GRUPO_DFP','DT_FIM_EXERC','DS_CONTA_PAI','DS_CONTA'], values='VL_CONTA', color_continuous_scale='Viridis', color='VL_CONTA', title='Treemap'):

  # https://plotly.com/python/treemaps/
  # https://plotly.com/python/network-graphs/

  # color_continuous_scale options
  #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
  #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
  #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |

  # Criar o Treemap utilizando a biblioteca Plotly Express
  fig = px.treemap(df,
                  path=path,
                  values=values,
                  #maxdepth=-1,
                  branchvalues='total',
                  color=color,
                  hover_data=['DT_FIM_EXERC'],
                  color_continuous_scale=color_continuous_scale,
                  )

  # Personalizar o layout do gráfico
  fig.update_layout(title=title, font=dict(size=12))
  fig.update_layout(margin = dict(t=50, l=0, r=0, b=0))

  # Adicionar interatividade para expandir e contrair os nós do Treemap
  fig.update_traces(textinfo="label+value", hovertemplate='<b>%{label}</b><br>Valor: R$ %{value}<br>Proporção: %{percentParent:.2%}<extra></extra>')

  # Exibir o gráfico
  st.plotly_chart(fig, use_container_width=True)
  
# ----------------------------------------------------------------------------------------------------

# Funções Sunburst

import plotly.express as px

# https://towardsdatascience.com/visualize-hierarchical-data-using-plotly-and-datapane-7e5abe2686e1

def gerar_sunburst(df, path=['GRUPO_DFP','DT_FIM_EXERC','DS_CONTA_PAI','DS_CONTA'], values='VL_CONTA', color_continuous_scale='Viridis', color='DS_CONTA', title='Sunburst'):

  fig = px.sunburst(df, 
                    path=path, 
                    values=values, 
                    color=color,
                    hover_data=['DT_FIM_EXERC'],
                    color_continuous_scale=color_continuous_scale,
                    )

  # Personalizar o layout do gráfico
  fig.update_layout(title_text=title, font_size=12)
  fig.update_layout(margin = dict(t=50, l=0, r=0, b=0))

  # Adicionar interatividade para expandir e contrair os nós do Treemap
  fig.update_traces(textinfo="label+value", hovertemplate='<b>%{label}</b><br>Valor: R$ %{value}<br>Proporção: %{percentParent:.2%}<extra></extra>')

  # Configuração do tamanho da figura
  fig.update_layout(
      width=800,  # Largura em pixels
      height=600  # Altura em pixels
  )

  # Exibir o gráfico
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------ 

# Funções NetworkX

from networkx import *
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.palettes import Spectral
import networkx as nx
import nx_altair as nxa

def desenha_grafico_rede(df, node_color='NIVEL_CONTA:N', cmap='viridis', title=''):

  # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html
  G = nx.from_pandas_edgelist(df, source='DS_CONTA_PAI', target='DS_CONTA', edge_attr=True, create_using=nx.Graph(), edge_key=None)
  pos = nx.circular_layout(G, scale=2, center=None, dim=2)
  pos = nx.kamada_kawai_layout(G)  # Utilizando o layout kamada-kawai para posicionar os nós

  # ----------------------------------------------------------------------------

  degrees = dict(G.degree(G.nodes()))
  nx.set_node_attributes(G, degrees, 'degree')  # Salva os graus como atributo dos nós

  between = nx.betweenness_centrality(G)
  nx.set_node_attributes(G, between, 'between')  # Salva a centralidade de intermediação como atributo dos nós

  # ----------------------------------------------------------------------------

  # Salva cada coluna do dataframe como um atributo do nó

  colunas = df.columns.to_list()
  for coluna in colunas:
    dict_atributo = dict(zip(df['DS_CONTA'], df[coluna]))
    nx.set_node_attributes(G, dict_atributo, coluna)

  # ----------------------------------------------------------------------------

  # Calcula o tamanho do nó com base em um percentual do tamanho do nó pai
  node_sizes = {}
  for node in G.nodes():
      if 'DS_CONTA_PAI' in G.nodes[node]:
          if G.nodes[node]['DS_CONTA_PAI'] != node:  # Verifica se não é o nó raiz
              parent = G.nodes[node]['DS_CONTA_PAI']
              if 'VL_CONTA' in G.nodes[parent]:
                  parent_size = G.nodes[parent]['VL_CONTA']
                  size = (G.nodes[node]['VL_CONTA'] / parent_size) * 1000
                  node_sizes[node] = size

  nx.set_node_attributes(G, node_sizes, 'node_size')  # Salva os tamanhos dos nós transformados

  # ----------------------------------------------------------------------------

  my_chart = nxa.draw_networkx(
      G=G,
      pos=pos,
      node_size='VL_CONTA:N',
      node_color=node_color,
      node_tooltip=["DT_REFER","DENOM_CIA","ORDEM_EXERC","CD_CONTA","DS_CONTA","ST_CONTA_FIXA:N","degree","between","VL_CONTA:Q","NIVEL_CONTA:N","DS_CONTA_PAI"],
      cmap=cmap,
      linewidths=0
  ).properties(
      title=title,
      width=800,
      height=600
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True) 
  
# ------------------------------------------------------------------------------ 


# ------------------------------------------------------------------------------ 

  
# ------------------------------------------------------------------------------ 

# Funções Plotly Network

# ------------------------------------------------------------------------------ 

import plotly.graph_objects as go
import networkx as nx

def desenha_grafico_rede_plotly(df, node_label=None, title=''):

  G = nx.from_pandas_edgelist(df, source='DS_CONTA_PAI', target='DS_CONTA', edge_attr=True, create_using=None, edge_key=None)
  pos = nx.kamada_kawai_layout(G) # Utilizando o layout kamada-kawai para posicionar os nós

# ------------------------------------------------------------------------------

  # Salva cada coluna do dataframe como um atributo do nó

  colunas = df.columns.to_list()
  for coluna in colunas:
    dict_atributo = dict(zip(df['DS_CONTA'], df[coluna]))
    nx.set_node_attributes(G, dict_atributo, coluna)

# ------------------------------------------------------------------------------

  # https://plotly.com/python/network-graphs/

  edge_x = []
  edge_y = []
  for edge in G.edges():
      x0, y0 = pos[edge[0]]
      x1, y1 = pos[edge[1]]
      edge_x.append(x0)
      edge_x.append(x1)
      edge_x.append(None)
      edge_y.append(y0)
      edge_y.append(y1)
      edge_y.append(None)

  edge_trace = go.Scatter(
      x=edge_x, y=edge_y,
      line=dict(width=0.5, color='#888'),
      hoverinfo='none',
      mode='lines')

  node_x = []
  node_y = []
  node_labels = []
  for node in G.nodes():
      x, y = pos[node]
      node_x.append(x)
      node_y.append(y)
      if 'DS_CONTA' in G.nodes[node]:
        node_labels.append(str(G.nodes[node]['DS_CONTA']))  # Use the 'DS_CONTA' attribute as the node label
      else:
        node_labels.append(str(node))  # Convert the label to a string if it's not already


  node_trace = go.Scatter(
      x=node_x, y=node_y,
      text=node_labels,  # Set the text to be displayed as the node label
      mode='markers+text',  # Use both markers and text for nodes
      hoverinfo='text',
      marker=dict(
          showscale=True,
          colorscale='YlGnBu',
          reversescale=True,
          color=[],
          size=10,
          colorbar=dict(
              thickness=15,
              title='Node Connections',
              xanchor='left',
              titleside='right'
          ),
          line_width=2),
      textposition='top center',  # Position the node label above the marker
      textfont=dict(color='black', size=12)  # Customize the node label font
  )

# ------------------------------------------------------------------------------

  node_adjacencies = []
  node_text = []
  for node, adjacencies in enumerate(G.adjacency()):
      node_adjacencies.append(len(adjacencies[1]))
      node_text.append('# of connections: '+str(len(adjacencies[1])))

  node_trace.marker.color = node_adjacencies
  node_trace.text = node_text

# ------------------------------------------------------------------------------

  if node_label:
    node_text = []
    for node in G.nodes():
        if 'DS_CONTA' in G.nodes[node]:
          node_text.append(str(G.nodes[node][node_label]))  # Use the 'node_label' attribute as the node label
        else:
          node_text.append(str(node))  # Convert the label to a string if it's not already

    node_trace.text = node_text

# ------------------------------------------------------------------------------

  fig = go.Figure(data=[edge_trace, node_trace],
              layout=go.Layout(
                  title='<br>' + title,
                  titlefont_size=16,
                  showlegend=False,
                  hovermode='closest',
                  margin=dict(b=20,l=5,r=5,t=40),
                  annotations=[ dict(
                      text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                      showarrow=False,
                      xref="paper", yref="paper",
                      x=0.005, y=-0.002 ) ],
                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                  )
  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------ 

import plotly.graph_objects as go

def gerar_waterfall(df, title='Waterfall'):

  fig = go.Figure()
  
  measures = []
  n = len(df)

  for i in range(n):
      if i == 0 or df['VL_CONTA'].iloc[i] < 0:
          measures.append("relative")
      elif i < n-1 and df['VL_CONTA'].iloc[i+1] > df['VL_CONTA'].iloc[i]:
          measures.append("relative")
      else:
          measures.append("total")

  fig.add_trace(go.Waterfall(
      name = "20",
      orientation = "v",
      measure = measures,
      x = df['DS_CONTA'],
      y = df['VL_CONTA'],
      textposition = "outside",
      text = df['VL_CONTA'],
      connector = {"line":{"color":"rgb(63, 63, 63)"}},
  ))

  fig.update_layout(
      title=title,
      xaxis=dict(title='Conta'),
      yaxis=dict(title='Valor (R$)'),
      showlegend = True,
      plot_bgcolor='rgba(0,0,0,0)',
      paper_bgcolor='rgba(0,0,0,0)',
      bargap=0.5,
  )

  fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

  fig.update_xaxes(showgrid=False)
  fig.update_yaxes(showgrid=True, gridcolor='lightgray')

  fig.update_layout(height=600, margin=dict(t=60, b=60, l=40, r=40))

  st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------ 
# ------------------------------------------------------------------------------ 

