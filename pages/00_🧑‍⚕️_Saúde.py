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

# ------------------------------------------------------------------------------

def plota_comparativo(df_csv, cd_cvm, Symbol):
  
  # ------------------------------------------------------------------------------
  
  datas_referencia = dfp.busca_datas_referencia(df_csv, cd_cvm)
  dt_refer = datas_referencia[0]
  # Buscar
  df_DFP = dfp.dados_da_empresa(df_csv, cd_cvm, dt_refer, 3)
  
  # Filtrar  
  df_DFP = fun.filtrar_df(df_DFP, criterios).sort_values(by=['DT_FIM_EXERC'])
  # Normalizar  
  vl_conta = df_DFP.loc[:, 'VL_CONTA']
  vl_conta_norm = vl_conta / vl_conta.iloc[0] * 100
  df_DFP = df_DFP.assign(VL_CONTA=vl_conta_norm)
    
  # ------------------------------------------------------------------------------  
  
  # https://altair-viz.github.io/gallery/layered_chart_with_dual_axis.html

  train_dataset = pd.read_csv("Quotes.csv")
  train_dataset = train_dataset[train_dataset['Date'] >= df_DFP.min(axis=0)['DT_FIM_EXERC']].reset_index()
  train_dataset.set_index("Date", inplace=True)
  train_dataset = train_dataset[train_dataset['Symbol'] == Symbol].reset_index()
    
  df_chart = train_dataset

  # Linha com o lucro
  grafico_linhas1 = alt.Chart(df_DFP).mark_line(color='green').encode(
    x=alt.X('DT_FIM_EXERC:T', title='Data'),
    y=alt.Y('VL_CONTA', title='Valor Conta'),
    tooltip=df_DFP.columns.to_list()
  )

  # Linha com a cotação de fechamento
  grafico_linhas2 = alt.Chart(train_dataset).mark_line(color='blue').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Adj_Close_Norm', title='Fechamento'),
    tooltip=train_dataset.columns.to_list()
  )

  my_chart = alt.layer(
    grafico_linhas1, grafico_linhas2
  ).resolve_scale(
    y='independent'
  ).properties(
    width=800,
    height=400,
    title={
    "text": f"Histórico de Preços do {Symbol} X Lucro ou Prejuízo Líquido Consolidado do Período",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  ).configure_axisX(
    labelAngle=0,
    tickCount=5,
    format='%d/%m/%Y'
  )

  # Legenda
  my_chart = my_chart.configure_legend(
    orient='bottom',
    labelFontSize=15,
    titleFontSize=15,
    title=None
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)
  
  df = dfp.pivotear_tabela(df_DFP).max().to_frame().T
  st.write(df)

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Saúde financeira? 🧑‍⚕️")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

# https://towardsdatascience.com/pagination-in-streamlit-82b62de9f62b
# https://docs.streamlit.io/library/api-reference/session-state
# https://docs.streamlit.io/library/api-reference/layout/st.columns

# Initialization
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 0

last_page = 2

# Adicionar os botões de anterior e próximo
prev, _ ,next = st.columns([1, 2, 3])

if next.button("Próxima"):

    if st.session_state.page_number + 1 > last_page:
        st.session_state.page_number = 0
    else:
        st.session_state.page_number += 1

if prev.button("Anterior"):

    if st.session_state.page_number - 1 < 0:
        st.session_state.page_number = last_page
    else:
        st.session_state.page_number -= 1

#st.write(f"Página: {st.session_state.page_number}")

# ------------------------------------------------------------------------------

if st.session_state.page_number == 0:

  st.header("O que é uma empresa saudável?")
  
  st.write("Resposta: Lucro. 🤑")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 1:

  st.header("Análise econômico-financeira")
  
  st.write("A análise será feita de acordo com o interesse do usuário da informação contábil (LIMEIRA, 2004, p. 78).")
  
  st.write("Os usuários internos e os externos são os dois grupos de pessoas que usam as demonstrações contábeis de uma empresa. Os usuários internos são os responsáveis pela gestão da empresa. Os prestadores de serviços, os bancos, os clientes, os concorrentes e outros são chamados de usuários externos (LIMEIRA, 2004, p. 78).")
  
  st.write("Basicamente, o objetivo da contabilidade por meio das demonstrações contábeis é fornecer informações para os público interno e externo para os processos de tomada de decisão, incluindo informações sobre a saúde financeira, a forma como a empresa capta recursos, a rentabilidade e o desempenho operacional da empresa (LIMEIRA, 2004, p. 78).")
  
  st.write("Os seguintes são os principais interessados na análise das demonstrações contábeis:")
  
  st.subheader("Sócios / acionistas")
  st.write("Examinam as demonstrações financeiras para obter informações sobre a solvência e a **lucratividade da empresa** 🤑. Eles também fazem comparações entre o desempenho da empresa em geral e o de anos anteriores (LIMEIRA, 2004, p. 78).")
  
  st.subheader("Fornecedores")
  st.write("Para proteger seus créditos 🤑 com mais segurança, eles devem conhecer a estrutura patrimonial de seus clientes, pois negociam o fornecimento de mercadorias e serviços, e dessa forma estão principalmente interessados na análise de liquidez (LIMEIRA, 2004, p. 78).")
  
  st.subheader("Instituições financeiras")
  st.write("Para proteger as operações de financiamento 🤑, desconto de duplicatas e outras operações financeiras, é importante conhecer a estrutura patrimonial da empresa (LIMEIRA, 2004, p. 78).")
  
  st.subheader("Clientes")
  st.write("Para garantir o fornecimento dos bens e serviços adquiridos 🤑, bem como o cumprimento dos prazos, os clientes devem estar cientes da situação financeira de seus fornecedores (LIMEIRA, 2004, p. 78).")
  
  st.subheader('"A avaliação de uma empresa é em parte arte, em parte ciência." (LOWE, 1998, p. 90)')
  
  st.write("**Referências:**")
  st.write("A L F LIMEIRA. Contabilidade para executivos. [s.l.] Rio De Janeiro; Editora Fgv, 2004.")
  st.write("LOWE, JANET. Warren Buffett: Dicas e Pensamentos do Maior Investidor do Mundo. Editora Elsevier, 1998.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 2:

  st.header("As cotações da empresa na bolsa de valores acompanham os lucros?")
  
  st.subheader('"Se a empresa vai bem, a ação pode imitar esse bom desempenho." (LOWE, 1998, p. 94)')
  
  #st.write("Demonstração do Resultado - Lucro ou Prejuízo Líquido Consolidado do Período")
  criterios = {}
  criterios['GRUPO_DFP'] = "Demonstração do Resultado"
  criterios['DS_CONTA_CLEAN'] = r'^(?=.*lucro)(?=.*prejuizo)(?=.*periodo)'
  #st.write(criterios)
  
  option = "dfp_cia_aberta_DRE_con.csv"
  df_csv = pd.read_csv(option)
  titulo = dfp.retorna_titulo(option)
  
  Symbol = "BBSE3.SA"
  cd_cvm = 23159 # BB SEGURIDADE PARTICIPAÇÕES S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)

  Symbol = "PSSA3.SA"
  cd_cvm = 16659 # PORTO SEGURO S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)
  
  Symbol = "ITUB4.SA"
  cd_cvm = 19348 # ITAU UNIBANCO HOLDING S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)
  
  Symbol = "SANB11.SA"
  cd_cvm = 20532 # BCO SANTANDER (BRASIL) S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)
  
  Symbol = "CIEL3.SA"
  cd_cvm = 21733 # CIELO S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)
  
  Symbol = "BRAP4.SA"
  cd_cvm = 18724 # BRADESPAR S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)
  
  Symbol = "BBDC4.SA"
  cd_cvm = 906 # BCO BRADESCO S.A.
  plota_comparativo(df_csv, cd_cvm, Symbol)
  
  Symbol = "B3SA3.SA"
  cd_cvm = 21610 # B3 S.A. - BRASIL, BOLSA, BALCÃO
  plota_comparativo(df_csv, cd_cvm, Symbol)

# ------------------------------------------------------------------------------
    
  st.write("**Referências:**")
  st.write("LOWE, JANET. Warren Buffett: Dicas e Pensamentos do Maior Investidor do Mundo. Editora Elsevier, 1998.")
      
# ------------------------------------------------------------------------------
