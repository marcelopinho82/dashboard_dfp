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

st.markdown("# Indicadores ❤️")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

# Definir a empresa a ser analisada
# https://discuss.streamlit.io/t/multiselect-widget-data-displayed-in-alphabetical-order/21617/2
denom_cia = st.selectbox('Qual a empresa gostaria de analisar?', dfp.lista_empresas())
cd_cvm = dfp.busca_cod_cvm_empresa(denom_cia)

tipo = st.selectbox('Demonstrativo:', ['Consolidado','Individual'])

# ------------------------------------------------------------------------------

df_concatenado = dfp.demonstrativos_empresa(cd_cvm)
df_concatenado = df_concatenado[(df_concatenado['GRUPO_DFP'].str.contains(tipo))]
df_concatenado = df_concatenado[(df_concatenado['CD_CONTA_PAI'].isin(["0","1","2","3","4","5"]))]

st.subheader("Dados brutos")
st.write(df_concatenado)

# ------------------------------------------------------------------------------

# https://www.statology.org/pandas-check-if-string-contains-multiple-substrings/
# https://analisemacro.com.br/economia/indicadores/automatizando-a-coleta-de-dados-de-demonstrativos-financeiros-como-comecar/

st.subheader("ROE - Retorno sobre o patrimônio (Return on equity)")

# Buscar os dados da primeira linha
st.write("Demonstração do Resultado - Lucro ou Prejuízo Líquido Consolidado do Período")
criterios1 = {}
criterios1['GRUPO_DFP'] = "Demonstração do Resultado"
criterios1['DS_CONTA_CLEAN'] = r'^(?=.*lucro)(?=.*prejuizo)(?=.*periodo)'
st.write(criterios1)

# Buscar os dados da segunda linha
st.write("Balanço Patrimonial Passivo - Patrimônio Líquido Consolidado")
criterios2 = {}
criterios2['GRUPO_DFP'] = "Balanço Patrimonial Passivo"
criterios2['DS_CONTA_CLEAN'] = r'^(?=.*patrimonio)(?=.*liquido)'
st.write(criterios2)

# Cálculo do indicador
df = fun.retorna_linhas(df_concatenado, criterios1, criterios2)
df_resultado = fun.calcula_indicador(df, 'ROE')
st.line_chart(df_resultado)
st.write(df_resultado.T)

# ------------------------------------------------------------------------------

st.subheader("Margem Líquida")

# Buscar os dados da primeira linha
st.write("Demonstração do Resultado - Lucro ou Prejuízo Líquido Consolidado do Período")
criterios1 = {}
criterios1['GRUPO_DFP'] = "Demonstração do Resultado"
criterios1['DS_CONTA_CLEAN'] = r'^(?=.*lucro)(?=.*prejuizo)(?=.*periodo)'
st.write(criterios1)

# Buscar os dados da segunda linha
st.write("Demonstração do Resultado - Receita de Venda de Bens e/ou Serviços")
criterios2 = {}
criterios2['GRUPO_DFP'] = "Demonstração do Resultado"
criterios2['DS_CONTA_CLEAN'] = r'^(?=.*receita)'
st.write(criterios2)

# Cálculo do indicador
df = fun.retorna_linhas(df_concatenado, criterios1, criterios2)
df_resultado = fun.calcula_indicador(df, 'Margem_Liq')
st.line_chart(df_resultado)
st.write(df_resultado.T)

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
