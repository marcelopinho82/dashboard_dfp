import os
import pandas as pd

def reduzir_dados(diretorio, lista_cd_cvm):
    # Percorre todos os arquivos .csv do diretório
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".csv"):
        
            print(arquivo)
        
            # Carrega o arquivo em um DataFrame
            caminho_arquivo = os.path.join(diretorio, arquivo)
            df = pd.read_csv(caminho_arquivo, low_memory=False)
            
            # Filtra as linhas com base na lista CD_CVM
            df_filtrado = df[df['CD_CVM'].isin(lista_cd_cvm)]
            
            # Salva o DataFrame filtrado no arquivo original
            df_filtrado.to_csv(caminho_arquivo, index=False)


diretorio = './'
lista_cd_cvm = [1023,2437,14966,15253,21733,15539,15709,16101,20044,16659] # CD_CVM

reduzir_dados(diretorio, lista_cd_cvm)
 
