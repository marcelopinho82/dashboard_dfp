#!/bin/bash

# https://www.b3.com.br/pt_br/market-data-e-indices/indices/acoes-por-indice/

# Nome do arquivo CSV
nome_arquivo="AcoesIndices_2023-06-29.csv"

# Codificação do arquivo
codificacao="ISO-8859-1"

# Separador utilizado no arquivo CSV
separador=";"

# Lê o arquivo CSV, descarta a primeira linha e obtém os códigos das empresas

empresas=$(tail -n +4 "$nome_arquivo" | iconv -f "$codificacao" -t UTF-8 | grep -v "BDRX" | grep -v "IFIX" | grep -v "IFIL" | cut -d "$separador" -f 1)
echo "$empresas" | sed 's/ S.A.//g' | sed 's/ S\/A//g'  > empresas.txt

codigos_empresas=$(tail -n +4 "$nome_arquivo" | iconv -f "$codificacao" -t UTF-8 | grep -v "BDRX" | grep -v "IFIX" | grep -v "IFIL" | cut -d "$separador" -f 3)
echo "$codigos_empresas" > tickers.txt

grep -Fwf empresas.txt cd_cvm_denom_cia.csv | cut -d "," -f 1 | tr '\n' ',' > cd_cvm.txt
