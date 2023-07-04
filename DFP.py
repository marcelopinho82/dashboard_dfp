import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------------------

# https://www.vooo.pro/insights/pivot-table-em-pandas-explicado/
# https://www.chesf.com.br/relainvest/Documents/Pages/CVM/DFP/DFP_DEZ2022%20CHESF.pdf
# https://www.easytweaks.com/pandas-pivot-table-sum/

def pivotear_tabela(df, index=['CD_CONTA', 'DS_CONTA'], aggfunc=np.sum, margins=False, margins_name='All'):
  pivot_table = pd.pivot_table(df, index=index, columns=['DT_FIM_EXERC'], values='VL_CONTA', aggfunc=aggfunc, fill_value=0, margins=margins, margins_name=margins_name)
  pivot_table = pivot_table.reset_index()
  pivot_table.index.name = None
  #import os
  #pivot_table.to_csv("Temp.csv", index=False)
  #pivot_table = pd.read_csv("Temp.csv", keep_default_na=False)
  #os.remove("Temp.csv")
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

import glob

def demonstrativos_empresa_na_data_referencia(cd_cvm):

  arquivos = glob.glob("*dfp_cia_aberta_*")

  # Inicializa uma lista vazia para armazenar os DataFrames carregados
  dfs = []

  # Itera sobre cada arquivo na lista "options"
  for arquivo in arquivos:

    # Carrega o arquivo em um DataFrame
    df_csv = pd.read_csv(arquivo, low_memory=False)
  
    # Definir a data de referência
    datas_referencia = busca_datas_referencia(df_csv, cd_cvm)
  
    if len(datas_referencia) > 0:
      dt_refer = datas_referencia[0]
  
      # Busca todos os dados da empresa na data de referência selecionada
      df_DFP = dados_da_empresa_na_data_referencia(df_csv, cd_cvm, dt_refer, 10)

      # Selecionar apenas o último exercício
      df_DFP = df_DFP[(df_DFP['ORDEM_EXERC'] == "ÚLTIMO")]

      # Adiciona o DataFrame à lista
      dfs.append(df_DFP)

  # Concatena todos os DataFrames na lista em um único DataFrame
  df_concatenado = pd.concat(dfs, ignore_index=True)
  df_concatenado.drop_duplicates(inplace=True)
  
  return df_concatenado

# ------------------------------------------------------------------------------

def demonstrativos_empresa(cd_cvm):

  arquivos = glob.glob("*dfp_cia_aberta_*")

  # Inicializa uma lista vazia para armazenar os DataFrames carregados
  dfs = []

  # Itera sobre cada arquivo na lista "options"
  for arquivo in arquivos:

    # Carrega o arquivo em um DataFrame
    df_csv = pd.read_csv(arquivo, low_memory=False)
  
    # Definir a data de referência
    datas_referencia = busca_datas_referencia(df_csv, cd_cvm)
  
    if len(datas_referencia) > 0:
      dt_refer = datas_referencia[0]
  
      # Busca todos os dados da empresa na data de referência selecionada
      df_DFP = dados_da_empresa(df_csv, cd_cvm, dt_refer, 10)

      # Adiciona o DataFrame à lista
      dfs.append(df_DFP)

  # Concatena todos os DataFrames na lista em um único DataFrame
  df_concatenado = pd.concat(dfs, ignore_index=True)
  df_concatenado.drop_duplicates(inplace=True)
  
  return df_concatenado

# ------------------------------------------------------------------------------

def busca_datas_referencia(df_ref, cd_cvm):
  datas_referencia = df_ref[(df_ref['CD_CVM'] == cd_cvm)]['DT_REFER'].unique() # Datas de referência disponíveis para a empresa selecionada
  #st.write(f"Datas de referência disponíveis para a empresa selecionada: {np.sort(datas_referencia)}")
  dt_refer = df_ref[(df_ref['CD_CVM'] == cd_cvm)]['DT_REFER'].max() # Utiliza a última data disponível como a data de referência
  #st.write(f"Última data de referência: {dt_refer}")
  return np.sort(datas_referencia)[::-1]

# ------------------------------------------------------------------------------

def transpor(df_ref):
  df = pivotear_tabela(df_ref)
  df = df.drop(['CD_CONTA'], axis=1)
  df.set_index("DS_CONTA", inplace=True)
  df = df.T
  return df

# ------------------------------------------------------------------------------

def analise_uma_conta(df_ref, CD_CONTA):
  df = pivotear_tabela(df_ref)
  df['NIVEL_CONTA'] = df["CD_CONTA"].str.count('\.') + 1
  NIVEL_CONTA = df[(df['CD_CONTA'] == CD_CONTA)]['NIVEL_CONTA'].unique()[0] + 1
  st.dataframe(df[df['CD_CONTA'] == CD_CONTA])
  df = df[df['CD_CONTA'].str.contains(CD_CONTA) & (df['NIVEL_CONTA'] == NIVEL_CONTA)]
  st.dataframe(df)
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
    
  elif option == "dfp_cia_aberta_BP_con.csv":
    titulo = "Balanço Patrimonial - Consolidado"

  elif option == "dfp_cia_aberta_BP_ind.csv":
    titulo = "Balanço Patrimonial - Individual"
    
  elif option == "dfp_cia_aberta_BP_DRE_con.csv":
    titulo = "Balanço Patrimonial + Demonstração de Resultado (DRE) - Consolidado"

  elif option == "dfp_cia_aberta_BP_DRE_ind.csv":
    titulo = "Balanço Patrimonial + Demonstração de Resultado (DRE) - Individual"

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
