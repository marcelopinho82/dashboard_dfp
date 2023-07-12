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

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Introdução 📖")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

# https://towardsdatascience.com/pagination-in-streamlit-82b62de9f62b
# https://docs.streamlit.io/library/api-reference/session-state
# https://docs.streamlit.io/library/api-reference/layout/st.columns

# Initialization
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 0

last_page = 15

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

  df = pd.read_csv("Tickers.csv")
  df.set_index("Date", inplace=True)

  st.write("Era uma vez um investidor que em Janeiro de 2017 decidiu investir em ações.")
  st.write("Ele já tinha algum conhecimento do mercado e decide escolher o setor financeiro para investir.")
  st.write("Ele sabia que o setor financeiro era um setor sólido com previsibilidade nos lucros e consulta as cotações da principais empresas do setor.")

  symbol = st.selectbox('Selecione o ticker:', df['Symbol'].unique(), key = "symbol")

  df_chart = df[df['Symbol'] == symbol].reset_index()

  # Criando o gráfico Altair
  my_chart = alt.Chart(df_chart).mark_bar().encode(
    x='Date:T',
    y='Low:Q',
    y2='High:Q',
    color=alt.condition("datum.Open <= datum.Close", alt.value("#06982d"), alt.value("#ae1325")),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Gráfico de Candlestick - {symbol}",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  # Regra Candlesticks
  my_chart += alt.Chart(df_chart).mark_rule().encode(
    x='Date:T',
    y='Open:Q',
    y2='Close:Q',
    color=alt.condition("datum.Open <= datum.Close", alt.value("#06982d"), alt.value("#ae1325"))
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 1:

  st.write("Depois de olhar os gráficos de Candlestick dos papéis que lhe interessaram, o investidor escolhe algumas empresas e decide comparar as cotações das empresas pelo preço de fechamento e pelo preço de fechamento ajustado. A diferença entre os dois gráficos é que o gráfico de preço de fechamento ajustado considera os efeitos de grupamentos, desdobramentos e outros eventos corporativos que influenciam no preço de fechamento.")
  
  df = pd.read_csv("Tickers.csv")
  df.set_index("Date", inplace=True)
  
  # Filtrar o dataframe pelos símbolos desejados
  tickers=['BBSE3.SA', 'PSSA3.SA', 'BBAS3.SA', 'ITSA4.SA']
  
  df_chart = df[df['Symbol'].isin(tickers)].reset_index()

  # Criando o gráfico com Altair
  my_chart = alt.Chart(df_chart).mark_line().encode(
    x='Date:T',
    y='Close:Q',
    color=alt.Color('Symbol:N', legend=alt.Legend(title='Ativo')),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Comparação de Preços de Fechamento - {', '.join(tickers)}",
    "fontSize": 16,
    "fontWeight": "bold"
    }
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
  
  df = pd.read_csv("Tickers.csv")
  df.set_index("Date", inplace=True)
  
  # Filtrar o dataframe pelos símbolos desejados
  tickers=['BBSE3.SA', 'PSSA3.SA', 'BBAS3.SA', 'ITSA4.SA']
  
  df_chart = df[df['Symbol'].isin(tickers)].reset_index()

  # Criando o gráfico com Altair
  my_chart = alt.Chart(df_chart).mark_line().encode(
    x='Date:T',
    y='Adj Close:Q',
    color=alt.Color('Symbol:N', legend=alt.Legend(title='Ativo')),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Comparação de Preços de Fechamento Ajustados - {', '.join(tickers)}",
    "fontSize": 16,
    "fontWeight": "bold"
    }
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

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 2:

  st.write("Não satisfeito com a comparação, o investidor decide comparar também pelo preço de fechamento normalizado, de modo a colocar na mesma escala os preços de fechamento das empresas.")

  df = pd.read_csv("Tickers.csv")
  df.set_index("Date", inplace=True)
  
  # Filtrar o dataframe pelos símbolos desejados
  tickers=['BBSE3.SA', 'PSSA3.SA', 'BBAS3.SA', 'ITSA4.SA']
  
  # Filtrar o dataframe pelos símbolos desejados
  df_chart = df[df['Symbol'].isin(tickers)].reset_index()

  # Criando o gráfico com Altair
  my_chart = alt.Chart(df_chart).mark_line().encode(
    x='Date:T',
    y='Adj_Close_Norm:Q',
    color=alt.Color('Symbol:N', legend=alt.Legend(title='Ativo')),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Comparação de Preços de Fechamento Normalizados - {', '.join(tickers)}",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  my_chart = my_chart.configure_legend(
    orient='bottom',
    labelFontSize=15,
    titleFontSize=15,
    title=None
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 3:

  st.header("💡")

  st.write("Lendo **as notícias da época** o investidor toma conhecimento que uma grande resseguradora vai abrir capital. É uma empresa bem conceituada que tem vários anos de história e sólida reputação no setor de seguros, que vai fazer um IPO (abertura de capital), através de uma oferta pública secundária das ações de emissão da companhia. Influenciado pelas matérias na mídia e identificando uma oportunidade o investidor decide se tornar acionista da empresa e participa do IPO reservando uma parcela do seu capital.")

  df = pd.read_csv("IRBR3_train.csv")

  st.write(f"Abaixo o gráfico de cotações do IRB entre os períodos de **{df.min(axis=0)['Date']}** e **{df.max(axis=0)['Date']}**.")

  df.set_index("Date", inplace=True)
  
  df_chart = df[df['Symbol'] == "IRBR3.SA"].reset_index()

  # Criando o gráfico Altair
  my_chart = alt.Chart(df_chart).mark_line(color='green').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Close', title='Fechamento'),
    tooltip=df_chart.columns.to_list()
  )

  # Adicionando a linha com a mínima
  my_chart += alt.Chart(df_chart).mark_line(color='red').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Low', title='Mínimo'),
    tooltip=df_chart.columns.to_list()
  )

  # Adicionando a linha com a máxima
  my_chart += alt.Chart(df_chart).mark_line(color='blue').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('High', title='Máxima'),
    tooltip=df_chart.columns.to_list()
  )

  # Adicionando a linha paralela onde o fechamento foi maior que a abertura
  my_chart += alt.Chart(df_chart[df_chart['Close'] > df_chart['Open']]).mark_line(color='blue').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Close', title='Fechamento'),
    color=alt.Color('Close:Q', legend=alt.Legend(title='Fechamento')),
    tooltip=df_chart.columns.to_list()
  )

  my_chart = my_chart.properties(
    width=800,
    height=600,
    title={
    "text": f"Histórico de Preços do IRBR3",
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

  st.write("As ações estreiam na B3 em 31 de julho de 2017 embutindo um múltiplo* de 1,3 o valor patrimonial.")
  st.write("Fonte: https://www.infomoney.com.br/mercados/irb-irbr3-veja-linha-do-tempo-desde-a-revelacao-de-inconsistencias-no-balanco-da-resseguradora/")
  st.write("Um múltiplo de mercado é a relação entre o preço do mercado de uma ação e suas variáveis operacionais, como dividendos, lucros, geração de caixa, entre outras. https://www.suno.com.br/artigos/multiplos-de-mercado/")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 4:

  st.header("😍")

  st.write("Tudo vai indo bem e o papel só sobe, **as notícias são todas positivas**.")

  st.write("Mesmo assim, em dúvida se fez um bom investimento, o investidor decide consultar o Oráculo.")
  
  st.write("O Oráculo é um homem muito sábio que estudou Ciência de Dados na UFRGS.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 5:

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"
  
  st.markdown("# O Oráculo 🔮")

  st.write("O investidor pede ao Oráculo para prever se o papel vai continuar subindo de modo que ele possa continuar a investir no ativo.")

  st.write(f"De posse dos dados que tinha até então, os preços de fechamento desde o IPO ocorrido em **{train_start_date}** até a data atual **{train_end_date}**, o oráculo faz a previsão do preço do ativo para os próximos 365 dias.")


  # Criando o gráfico Altair
  my_chart = alt.Chart(forecast).mark_line(color='blue').encode(
    x=alt.X('ds:T', title='Data'),
    y=alt.Y('yhat', title='Preço de fechamento'),
    tooltip=forecast.columns.to_list()
  )

  # Adicionando a banda de variação
  my_chart += alt.Chart(forecast[forecast['ds'] > train_end_date]).mark_area(opacity=0.2, color='gray').encode(
    x='ds:T',
    y=alt.Y('yhat_lower:Q', title='Banda Inferior'),
    y2=alt.Y2('yhat_upper:Q', title='Banda Superior'),
  )

  my_chart = my_chart.properties(
    width=800,
    height=700,
    title='Previsão de preço da IRBR3'
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

  st.write("O Oráculo lhe diz para ficar tranquilo porque o papel vai continuar subindo.")

  st.write("Inclusive ele lhe mostra a clara tendência de alta do papel ao analisar a série temporal.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 6:

  st.header("🤑")

  st.write("Então o investidor tranquilo e confiante no que o Oráculo lhe disse vai 'all in' e aposta tudo no papel.")

  st.write("Afinal, ele pensa, não há como o Oráculo estar errado.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 7:

  st.header("⏳")

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"

  st.write("Algum tempo depois ...")

  st.write("O investidor decide comparar o que o Oráculo preveu com a realidade.")

  st.write(f"Ele compara as cotações reais com as cotações previstas no período de **{test_start_date}** até **{test_end_date}**.")
  
  # Criando o gráfico de linha dos dados de treinamento
  my_chart = alt.Chart(train_dataset[train_dataset['Symbol'] == "IRBR3.SA"].reset_index()).mark_line(color='black').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Close:Q', title='Valor Real Treinamento'),
    tooltip=train_dataset.columns.to_list()
  )

  # Adicionando a linha dos dados de teste
  my_chart += alt.Chart(test_dataset[test_dataset['Symbol'] == "IRBR3.SA"].reset_index()).mark_line(color='lightgray').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Close:Q', title='Valor Real Teste'),
    tooltip=test_dataset.columns.to_list()
  )

  # Adicionando a linha dos dados previstos
  my_chart += alt.Chart(forecast).mark_line(color='blue').encode(
    x=alt.X('ds:T', title='Data'),
    y=alt.Y('yhat:Q', title='Valor Previsto'),
    tooltip=['yhat']
  )

  # Adicionando a banda de variação
  my_chart += alt.Chart(forecast[forecast['ds'] > train_end_date]).mark_area(opacity=0.2, color='gray').encode(
    x='ds:T',
    y=alt.Y('yhat_lower:Q', title='Banda Inferior'),
    y2=alt.Y2('yhat_upper:Q', title='Banda Superior'),
  )

  my_chart = my_chart.properties(
    width=800,
    height=700,
    title={
    "text": f"Previsão X Realidade - IRBR3",
    "fontSize": 16,
    "fontWeight": "bold"
    }
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

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 8:

  st.header("😡")

  st.write("O investidor fica em choque ao perceber que o Oráculo errou. Ele não entende como isso pode ter acontecido e decide investigar o que aconteceu.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 9:

  st.header("🙈🙉🙊")

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"

  st.write(f"O investidor olha novamente o gráfico de Candlestick no período posterior a previsão do oráculo **({test_start_date})** até a data final da cotação de fechamento que possui **({test_end_date})**...")

  st.write(f"Abaixo o gráfico de cotações do IRB entre os períodos de **{train_start_date}** e **{test_end_date}**.")

  df_chart = pd.concat([train_dataset[(train_dataset['Symbol'] == "IRBR3.SA")].reset_index(), test_dataset[(test_dataset['Symbol'] == "IRBR3.SA")].reset_index()])

  # Criando o gráfico Altair
  my_chart = alt.Chart(df_chart).mark_bar().encode(
    x='Date:T',
    y='Low:Q',
    y2='High:Q',
    color=alt.condition("datum.Open <= datum.Close", alt.value("#06982d"), alt.value("#ae1325")),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Gráfico de Candlestick - Previsão X Realidade - IRBR3",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  # Regra Candlesticks
  my_chart += alt.Chart(df_chart).mark_rule().encode(
    x='Date:T',
    y='Open:Q',
    y2='Close:Q',
    color=alt.condition("datum.Open <= datum.Close", alt.value("#06982d"), alt.value("#ae1325"))
  )

  # Adicionando a linha dos dados previstos
  my_chart += alt.Chart(forecast).mark_line(color='blue').encode(
    x=alt.X('ds:T', title='Data'),
    y=alt.Y('yhat:Q', title='Valor Previsto'),
    tooltip=['yhat']
  )

  # Adicionando a banda de variação
  my_chart += alt.Chart(forecast[forecast['ds'] > train_end_date]).mark_area(opacity=0.2, color='gray').encode(
    x='ds:T',
    y=alt.Y('yhat_lower:Q', title='Banda Inferior'),
    y2=alt.Y2('yhat_upper:Q', title='Banda Superior'),
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)
  
  st.write(f"Olhando o gráfico, o investidor nota uma queda brusca na cotação do ativo a partir da data de **{train_end_date}**.")
  st.write("Os sucessivos candles vermelhos de baixa sustentam uma queda que depreciam sobremaneira as cotações e invertem drasticamente a tendência.")
  
# ------------------------------------------------------------------------------
  
elif st.session_state.page_number == 10:

  st.header("😩")

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"
  
  st.write(f"Abaixo o gráfico de volume do IRB entre os períodos de **{train_start_date}** e **{test_end_date}**.")

  df_chart = pd.concat([train_dataset[train_dataset['Symbol'] == "IRBR3.SA"].reset_index(), test_dataset[test_dataset['Symbol'] == "IRBR3.SA"].reset_index()])

  # Criando o gráfico com Altair
  my_chart = alt.Chart(df_chart).mark_line(color='blue').encode(
    x=alt.Y('Date:T', title='Data'),
    y=alt.Y('Volume:Q', title='Volume'),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Volume Negociado - IRBR3",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)

  max_variation=df_chart['Volume'].values.argmax()
  max_value=df_chart['Volume'].values[max_variation,]
  st.dataframe(df_chart[df_chart['Volume'].values == max_value])
  
  st.write("O gráfico de volume no período analisado indica uma forte pressão vendedora sobre o papel.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 11:

  st.header("💸")

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"
  
  st.write(f"Abaixo o gráfico de variação de preço do IRB entre os períodos de **{train_start_date}** e **{test_end_date}**.")

  df_chart = pd.concat([train_dataset[train_dataset['Symbol'] == "IRBR3.SA"].reset_index(), test_dataset[test_dataset['Symbol'] == "IRBR3.SA"].reset_index()])

  # Criando o gráfico com Altair
  my_chart = alt.Chart(df_chart).mark_line().encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Variation:Q', title='Variação de Preço'),
    tooltip=df_chart.columns.to_list()
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Variação de Preço Diário (Antes + Depois) - IRBR3",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  # Adicionando a linha onde o nível de variação foi maior que 1
  my_chart += alt.Chart(df_chart[df_chart['Variation'] > 1]).mark_line(color='blue').encode(
    x=alt.X('Date:T', title='Data'),
    y=alt.Y('Variation:Q', title='Variação de Preço'),
    color=alt.Color('Variation:Q', legend=alt.Legend(title='Variação')),
    tooltip=df_chart.columns.to_list()
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

  max_variation=df_chart['Variation'].values.argmax()
  max_value=df_chart['Variation'].values[max_variation,]
  st.dataframe(df_chart[df_chart['Variation'].values == max_value])
  
  st.write("O gráfico de variação do preço diário no período analisado indica uma forte variação no preço do papel após a data prevista pelo Oráculo.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 12:

  st.header("😳")

  st.write("Não satisfeito, ele decide fazer um histograma com os preços de fechamento antes e depois para verificar a variação nos preços.")

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"

  df_chart = train_dataset[train_dataset['Symbol'] == "IRBR3.SA"].reset_index()

  # Criando o gráfico Altair
  my_chart = alt.Chart(df_chart).transform_density(
    'Close',
    as_=['Close', 'density']
  ).mark_bar(
    color='green',
    opacity=0.5,
  ).encode(
    x=alt.X('Close:Q', title='Fechamento'),
    y=alt.Y('density:Q', axis=alt.Axis(title='Density')),
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Distribuição Preço Fechamento (Antes) - IRBR3",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 13:

  st.header("😟")

  train_dataset = pd.read_csv("IRBR3_train.csv")
  train_dataset.set_index("Date", inplace=True)
  test_dataset = pd.read_csv("IRBR3_test.csv")
  test_dataset.set_index("Date", inplace=True)
  forecast = pd.read_csv("IRBR3_forecast.csv")
  forecast['ds'] = pd.to_datetime(forecast['ds'].astype(str), format='%Y-%m-%d')
  train_start_date="2017-07-31"
  train_end_date="2020-02-02"
  test_start_date="2020-02-03"
  test_end_date="2021-02-01"

  df_chart = test_dataset[test_dataset['Symbol'] == "IRBR3.SA"].reset_index()

  # Criando o gráfico Altair
  my_chart = alt.Chart(df_chart).transform_density(
    'Close',
    as_=['Close', 'density']
  ).mark_bar(
    color='green',
    opacity=0.5,
  ).encode(
    x=alt.X('Close:Q', title='Fechamento'),
    y=alt.Y('density:Q', axis=alt.Axis(title='Density')),
  ).properties(
    width=800,
    height=600,
    title={
    "text": f"Distribuição Preço Fechamento (Depois) - IRBR3",
    "fontSize": 16,
    "fontWeight": "bold"
    }
  )

  # Exibir o gráfico interativo
  st.altair_chart(my_chart.interactive(), use_container_width=True)

  st.write("Nosso investidor percebe que as cotações de fechamento antes e depois variaram drasticamente.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 14:

  st.header("😢")

  st.write("Depois da decepção ... Decidido a não ser mais parte da manada, nosso herói decide embarcar uma jornada de auto-conhecimento para ser capaz de realizar as próprias análises sem depender dos conselhos de um Oráculo.")
  st.write("Ele aprendeu que rentabilidade passada não é garantia de retorno futuro e sabe que para escolher uma boa empresa é necessário analisar os seus fundamentos e não o gráficos das cotações da empresa na bolsa.")
  st.write("Infelizmente, lá no fundo, o que esse pobre investidor não sabia é que a empresa em que ele investiu todo o seu dinheiro foi vítima de uma fraude contábil.")
  st.write("Fim da história.")
  st.write("")
  st.header("**Moral da história:**")
  st.write("Nenhum modelo de previsão de machine learning poderia ser capaz de prever a derrocada das ações da empresa.")
  st.write("Somente o acompanhamento do balanço da empresa, ou seja, as DFP disponibilizadas no site da CVM, através de uma análise minuciosa, poderia ser capaz de prever o cenário que se concretizou.")

# ------------------------------------------------------------------------------

elif st.session_state.page_number == 15:

  st.header("❓")

  st.write("As perguntas que ficam são:")

  st.write("* Isso foi um caso isolado? ou isso já ocorreu antes?")
  st.write("* Qual o papel das notícias nessa história toda? Como as notícias influenciam as decisões de investimento? Fica óbvio que as notícias tiveram um papel fundamental na queda dos preços nas cotações da empresa na bolsa, mas será que isso é algo que sempre acontece? Qual o verdadeiro impacto?")
  st.subheader("* **É possível utilizar a visualização de dados para prever ou entender melhor a saúde financeira de uma empresa?**")

# ------------------------------------------------------------------------------
