import pandas as pd
import numpy as np
import altair as alt
#import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------

import altair as alt

# https://stackoverflow.com/questions/72181211/grouped-bar-charts-in-altair-using-two-different-columns

# ------------------------------------------------------------------------------

import pandas as pd
import altair as alt

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
      color=alt.Color('ORDEM_EXERC:N', legend=alt.Legend(title='Exercício'), scale=alt.Scale(domain=retorna_colunas_data(df), range=retorna_cores(len(retorna_colunas_data(df))))),
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
  #st.altair_chart(my_chart.interactive(), use_container_width=True)
  display(my_chart.interactive())

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
  #st.altair_chart(my_chart.interactive(), use_container_width=True)
  display(my_chart.interactive())

# ------------------------------------------------------------------------------

def grafico_3(df, titulo, denom_cia, dt_refer):

  #niveis = list(df['NIVEL_CONTA'].unique())
  #niveis.sort(reverse=False)

  #filtro_nivel_conta = alt.selection_single(
  #    name='Nivel',
  #    fields=['NIVEL_CONTA'],
  #    bind=alt.binding_select(options=niveis),
  #    init={'NIVEL_CONTA': min(niveis)}
  #)

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
  )#.add_selection(filtro_nivel_conta).transform_filter(filtro_nivel_conta)

  # Exibir o gráfico interativo
  #st.altair_chart(my_chart.interactive(), use_container_width=True)
  display(my_chart.interactive())

# ------------------------------------------------------------------------------

# https://www.vooo.pro/insights/pivot-table-em-pandas-explicado/
# https://www.chesf.com.br/relainvest/Documents/Pages/CVM/DFP/DFP_DEZ2022%20CHESF.pdf
# https://www.easytweaks.com/pandas-pivot-table-sum/

def pivotear_tabela(df, index=['CD_CONTA', 'DS_CONTA'], margins=False, margins_name='All'):
  pivot_table = pd.pivot_table(df, index=index, columns=['DT_FIM_EXERC'], values='VL_CONTA', aggfunc=np.sum, fill_value=0, margins=margins, margins_name=margins_name)
  pivot_table = pivot_table.reset_index()

  import os
  pivot_table.to_csv("Temp.csv", index=False)
  pivot_table = pd.read_csv("Temp.csv", keep_default_na=False)
  os.remove("Temp.csv")

  return pivot_table

# ------------------------------------------------------------------------------

def dados_da_empresa(df_ref, cd_cvm, dt_refer, nivel_conta):
  df = df_ref[(df_ref['CD_CVM'] == cd_cvm) & (df_ref['VL_CONTA'] != 0) & (df_ref['NIVEL_CONTA'] <= nivel_conta) & (df_ref['DT_REFER'] <= dt_refer) & (df_ref['ORDEM_EXERC'] == "ÚLTIMO")]
  return df

# ------------------------------------------------------------------------------

def dados_da_empresa_na_data_referencia(df_ref, cd_cvm, dt_refer, nivel_conta):
  df = df_ref[(df_ref['CD_CVM'] == cd_cvm) & (df_ref['VL_CONTA'] != 0) & (df_ref['NIVEL_CONTA'] <= nivel_conta) & (df_ref['DT_REFER'] == dt_refer)]
  return df

# ------------------------------------------------------------------------------

def busca_datas_referencia(df_ref, cd_cvm):
  datas_referencia = df_ref[(df_ref['CD_CVM'] == cd_cvm)]['DT_REFER'].unique() # Datas de referência disponíveis para a empresa selecionada
  print(f"Datas de referência disponíveis para a empresa selecionada: {np.sort(datas_referencia)}")
  dt_refer = df_ref[(df_ref['CD_CVM'] == cd_cvm)]['DT_REFER'].max() # Utiliza a última data disponível como a data de referência
  print(f"Última data de referência: {dt_refer}")
  return np.sort(datas_referencia)[::-1]

# ------------------------------------------------------------------------------

def grafico_comparativo_duas_contas(df_ref, titulo, denom_cia, dt_refer, cd_conta_1, cd_conta_2):

  df1 = df_ref[(df_ref['CD_CONTA'] == cd_conta_1)]
  df2 = df_ref[(df_ref['CD_CONTA'] == cd_conta_2)]
  df = pd.concat([df1, df2])
  display(pivotear_tabela(df, margins=True, margins_name='Total'))

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
  #st.altair_chart(my_chart.interactive(), use_container_width=True)
  display(my_chart.interactive())

# ------------------------------------------------------------------------------

def retorna_colunas_data(df):

  # https://stackoverflow.com/questions/3640359/regular-expressions-search-in-list
  import re
  mylist = df.columns.to_list()
  r = re.compile(r'\d{1,4}-\d{1,2}-\d{1,2}')
  newlist = list(filter(r.match, mylist)) # Read Note below

  return newlist

# ------------------------------------------------------------------------------

def transpor(df):

  import os

  df2 = df.transpose()
  df2.to_csv("Temp.csv", index=False)
  df2 = pd.read_csv("Temp.csv")
  os.remove("Temp.csv")

  return df2

# ------------------------------------------------------------------------------

def retorna_cores(x):
  # https://colordesigner.io/gradient-generator -
  # 15 cores no total
  cores = ['#fafa6e','#d7f171','#b5e877','#95dd7d','#77d183','#5bc489','#3fb78d','#23aa8f','#009c8f','#008d8c','#007f86','#0b717e','#1c6373','#255566','#2a4858']
  return cores[0:x]

def retorna_cores_2(x):
  cores = ['#d7d6fe','#c5c3ff','#b3afff','#a19cff','#9088ff','#7d74ff','#6a5fff','#5549ff','#3b2fff','#0c00ff']
  return cores[0:x]

# ------------------------------------------------------------------------------

def analise_uma_conta(df_ref, CD_CONTA):
  df = pivotear_tabela(df_ref)
  df['NIVEL_CONTA'] = df["CD_CONTA"].str.count('\.') + 1
  NIVEL_CONTA = df[(df['CD_CONTA'] == CD_CONTA)]['NIVEL_CONTA'].unique()[0] + 1
  display(df[df['CD_CONTA'] == CD_CONTA])
  df = df[df['CD_CONTA'].str.contains(CD_CONTA) & (df['NIVEL_CONTA'] == NIVEL_CONTA)]
  display(df)
  return df

# ------------------------------------------------------------------------------

def retorna_titulo(option):

  if option == "dfp_cia_aberta_BPA_con.csv":
    titulo = "Balanço Patrimonial Ativo (BPA) - Consolidado"

  elif option == "dfp_cia_aberta_BPA_ind.csv":
    titulo = "Balanço Patrimonial Ativo (BPA) - Individual"

  elif option == "dfp_cia_aberta_BPP_con.csv":
    titulo = "Balanço Patrimonial Passivo (BPP) - Consolidado"

  elif option == "dfp_cia_aberta_BPP_ind.csv":
    titulo = "Balanço Patrimonial Passivo (BPP) - Individual"

  elif option == "dfp_cia_aberta_DFC_MD_con.csv":
    titulo = "Demonstração de Fluxo de Caixa - Método Direto (DFC-MD) - Consolidado"

  elif option == "dfp_cia_aberta_DFC_MD_ind.csv":
    titulo = "Demonstração de Fluxo de Caixa - Método Direto (DFC-MD) - Individual"

  elif option == "dfp_cia_aberta_DFC_MI_con.csv":
    titulo = "Demonstração de Fluxo de Caixa - Método Indireto (DFC-MI) - Consolidado"

  elif option == "dfp_cia_aberta_DFC_MI_ind.csv":
    titulo = "Demonstração de Fluxo de Caixa - Método Indireto (DFC-MI) - Individual"

  elif option == "dfp_cia_aberta_DMPL_con.csv":
    titulo = "Demonstração das Mutações do Patrimônio Líquido (DMPL) - Consolidado"

  elif option == "dfp_cia_aberta_DMPL_ind.csv":
    titulo = "Demonstração das Mutações do Patrimônio Líquido (DMPL) - Individual"

  elif option == "dfp_cia_aberta_DRA_con.csv":
    titulo = "Demonstração de Resultado Abrangente (DRA) - Consolidado"

  elif option == "dfp_cia_aberta_DRA_ind.csv":
    titulo = "Demonstração de Resultado Abrangente (DRA) - Individual"

  elif option == "dfp_cia_aberta_DRE_con.csv":
    titulo = "Demonstração de Resultado (DRE) - Consolidado"

  elif option == "dfp_cia_aberta_DRE_ind.csv":
    titulo = "Demonstração de Resultado (DRE) - Individual"

  elif option == "dfp_cia_aberta_DVA_con.csv":
    titulo = "Demonstração de Valor Adicionado (DVA) - Consolidado"

  elif option == "dfp_cia_aberta_DVA_ind.csv":
    titulo = "Demonstração de Valor Adicionado (DVA) - Individual"

  else:
    titulo = ""

  return titulo

# ------------------------------------------------------------------------------

import glob

def nome_arquivo(padrao):
  arquivos = glob.glob("*" + padrao + "*.csv")
  nome_arquivo = arquivos[0]
  return nome_arquivo

# ------------------------------------------------------------------------------

import gc

def busca_cod_cvm_empresa(denom_cia):
  df_csv = pd.read_csv("cd_cvm_denom_cia.csv", encoding="utf-8", sep=',')
  df = df_csv[df_csv['DENOM_CIA'].str.contains(denom_cia)][['CD_CVM','DENOM_CIA']].drop_duplicates()
  del df_csv
  gc.collect()
  return df.iloc[0][0]

# ------------------------------------------------------------------------------

def lista_empresas():
  df_csv = pd.read_csv("cd_cvm_denom_cia.csv", encoding="utf-8", sep=',')
  df = df_csv
  lista_empresas = df.DENOM_CIA.unique().tolist()
  del df_csv
  del df
  gc.collect()
  return lista_empresas

# ------------------------------------------------------------------------------
