# ------------------------------------------------------------------------------

import pandas_datareader.data as data  # Submódulo do pandas_datareader para coleta de dados financeiros
import yfinance as yf  # Biblioteca para coleta de dados financeiros da plataforma Yahoo! Finance
# https://stackoverflow.com/questions/74912452/typeerror-string-indices-must-be-integer-pandas-datareader
yf.pdr_override()  # Configuração para sobrepor a função de coleta de dados padrão do pandas_datareader pelo Yahoo! Finance

train_start_date="2016-01-01"
train_end_date="2017-01-15"

# https://blog.toroinvestimentos.com.br/bolsa/acoes-de-seguradoras
# https://investidor10.com.br/setores/financeiro/
# https://euqueroinvestir.com/educacao-financeira/vale-a-pena-investir-em-empresas-do-setor-financeiro
tickers=['BBSE3.SA', 'PSSA3.SA', 'ITUB4.SA', 'SANB11.SA', 'ITSA4.SA', 'CIEL3.SA', 'BRAP4.SA', 'BBDC4.SA', 'BBDC3.SA', 'B3SA3.SA']

import pandas as pd
import numpy as np

train_dataset = pd.DataFrame()
test_dataset = pd.DataFrame()

for ticker in tickers:

  print(ticker)

  train_dataset_symbol = data.get_data_yahoo(ticker, start = train_start_date, end = train_end_date)
  train_dataset_symbol['Symbol'] = ticker
  train_dataset_symbol['Variation'] = abs(train_dataset_symbol['Close'] - train_dataset_symbol['Open'])
  adj_close = train_dataset_symbol.loc[:, 'Adj Close']
  adj_close_norm = adj_close / adj_close.iloc[0] * 100
  train_dataset_symbol = train_dataset_symbol.assign(Adj_Close_Norm=adj_close_norm)
  train_dataset_symbol['Log_Returns'] = np.log(train_dataset_symbol['Adj_Close_Norm'] / train_dataset_symbol['Adj_Close_Norm'].shift(1))
  train_dataset = pd.concat([train_dataset, train_dataset_symbol])

train_dataset.to_csv("Tickers" + ".csv", encoding='utf-8', index=True)

# ------------------------------------------------------------------------------

# https://www.infomoney.com.br/mercados/irb-irbr3-veja-linha-do-tempo-desde-a-revelacao-de-inconsistencias-no-balanco-da-resseguradora/
train_start_date="2017-07-31"
train_end_date="2020-02-02"

test_start_date="2020-02-03"
test_end_date="2021-02-01"

tickers=['IRBR3.SA'] 

train_dataset = pd.DataFrame()
test_dataset = pd.DataFrame()

for ticker in tickers:

  print(ticker)

  train_dataset_symbol = data.get_data_yahoo(ticker, start = train_start_date, end = train_end_date)
  train_dataset_symbol['Symbol'] = ticker
  train_dataset_symbol['Variation'] = abs(train_dataset_symbol['Close'] - train_dataset_symbol['Open'])
  adj_close = train_dataset_symbol.loc[:, 'Adj Close']
  adj_close_norm = adj_close / adj_close.iloc[0] * 100
  train_dataset_symbol = train_dataset_symbol.assign(Adj_Close_Norm=adj_close_norm)
  train_dataset_symbol['Log_Returns'] = np.log(train_dataset_symbol['Adj_Close_Norm'] / train_dataset_symbol['Adj_Close_Norm'].shift(1))
  train_dataset = pd.concat([train_dataset, train_dataset_symbol])

  test_dataset_symbol = data.get_data_yahoo(ticker, start = test_start_date, end = test_end_date)
  test_dataset_symbol['Symbol'] = ticker
  test_dataset_symbol['Variation'] = abs(test_dataset_symbol['Close'] - test_dataset_symbol['Open'])
  adj_close = test_dataset_symbol.loc[:, 'Adj Close']
  adj_close_norm = adj_close / adj_close.iloc[0] * 100
  test_dataset_symbol = test_dataset_symbol.assign(Adj_Close_Norm=adj_close_norm)
  test_dataset_symbol['Log_Returns'] = np.log(test_dataset_symbol['Adj_Close_Norm'] / test_dataset_symbol['Adj_Close_Norm'].shift(1))
  test_dataset = pd.concat([test_dataset, test_dataset_symbol])
  
train_dataset.to_csv("IRBR3_" + "train" + ".csv", encoding='utf-8', index=True)
test_dataset.to_csv("IRBR3_" + "test" + ".csv", encoding='utf-8', index=True)

# ------------------------------------------------------------------------------
