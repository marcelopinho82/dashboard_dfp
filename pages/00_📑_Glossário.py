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

def main():

    glossario = {
        "Balanço Patrimonial Ativo (BPA)": 
        "O Balanço Patrimonial Ativo representa o Relatório Financeiro (contábil) com o objetivo de apresentar, de forma quantitativa e qualitativa, a partir de uma data específica, a situação monetária da organização. Ele apresenta os bens e direitos controlados pela empresa em determinado período (RIBEIRO, 2017a, p. 402).",
        
        "Balanço Patrimonial Passivo (BPP)": 
        "O Balanço Patrimonial Passivo representa o Relatório Financeiro (contábil) com o objetivo de apresentar, de forma quantitativa e qualitativa, a partir de uma data específica, a situação monetária da organização. Ele apresenta as obrigações e dívidas, bem como o patrimônio líquido da empresa em uma data especificada (RIBEIRO, 2017a, p. 402).",
        
        "Demonstração de Fluxo de Caixa - Método Direto (DFC-MD)":
        "A Demonstração dos Fluxos de Caixa (DFC) é um documento contábil que se esforça para revelar as transações que ocorreram em uma duração especificada e resultaram em alterações na posição de caixa e nos equivalentes de caixa.<br><br>É uma declaração sucinta das realidades gerenciais que envolvem os movimentos de caixa que ocorreram durante um período específico, documentada com precisão em débito (entradas) e crédito (saídas) da conta de caixa, da conta de movimentação de contas bancárias e das contas que representam equivalentes de caixa (RIBEIRO, 2017a, p. 431).<br><br>Assim, a Demonstração de Fluxo de Caixa pelo Método Direto apresenta os valores reais de entrada e saída de dinheiro durante um período, identificando a origem e o destino dos recursos.",
        
        "Demonstração de Fluxo de Caixa - Método Indireto (DFC-MI)":        
        "A Demonstração dos Fluxos de Caixa (DFC) é um documento contábil que se esforça para revelar as transações que ocorreram em uma duração especificada e resultaram em alterações na posição de caixa e nos equivalentes de caixa.<br><br>É uma declaração sucinta das realidades gerenciais que envolvem os movimentos de caixa que ocorreram durante um período específico, documentada com precisão em débito (entradas) e crédito (saídas) da conta de caixa, da conta de movimentação de contas bancárias e das contas que representam equivalentes de caixa (RIBEIRO, 2017a, p. 431).<br><br>Assim, a Demonstração de Fluxo de Caixa pelo Método Indireto começa com o lucro líquido e ajusta esse valor para obter o fluxo de caixa operacional.",
        
        "Demonstração das Mutações do Patrimônio Líquido (DMPL)": 
        "A Demonstração das Mutações do Patrimônio Líquido (DMPL) é um relatório contábil que visa a evidenciar as variações ocorridas em todas as contas que compõem o Patrimônio Líquido em um determinado período (RIBEIRO, 2017a, p. 427).<br><br>Assim, a Demonstração das Mutações do Patrimônio Líquido apresenta as alterações ocorridas no patrimônio líquido da empresa ao longo de um período, incluindo lucros, prejuízos, distribuição de dividendos, entre outros.",
        
        "Demonstração de Resultado Abrangente (DRA)":        
        "A adoção dos padrões internacionais de contabilidade no Brasil trouxe a obrigatoriedade da divulgação da Demonstração do Resultado Abrangente. Sem dúvida nenhuma, foi um avanço o reconhecimento de variações patrimoniais que ainda não transitaram pelo resultado. No entanto, do ponto de vista da tomada de decisões, ainda é preciso amadurecer o uso da referida demonstração. Não está consolidado na literatura, por exemplo, o efeito dos resultados abrangentes nos índices de rentabilidade, embora esteja claro que os valores nela reconhecidos são uma prévia de resultados futuros.<br><br>A Demonstração do Resultado Abrangente do Exercício apresenta as receitas, despesas e outras mutações que afetam o Patrimônio Líquido que não foram reconhecidas na Demonstração do Resultado do Exercício, conforme determina o Pronunciamento CPC 26. De acordo com o referido pronunciamento, os outros resultados abrangentes compreendem:<br><br>a) Variações na reserva de reavaliação quando permitidas legalmente.<br><br>b) Ganhos e perdas atuariais em planos de pensão (benefícios a empregados).<br><br>c) Ganhos e perdas derivados de conversão de demonstrações contábeis de operações no exterior.<br><br>d) Ajuste de avaliação patrimonial relativo aos ganhos e perdas na remensuração de ativos financeiros disponíveis para venda.<br><br>e) Ajuste de avaliação patrimonial relativo à efetiva parcela de ganhos ou perdas de instrumentos de hedge em hedge de fluxo de caixa e outros.<br><br>A DRA é apresentada em um relatório próprio, tendo como valor inicial (a) o lucro líquido do exercício, (b) seguido dos resultados abrangentes, (c) dos resultados abrangentes de empresas investidas reconhecidos por meio da equivalência patrimonial e (d) o resultado abrangente do período (MARTINS; MIRANDA; DINIZ, 2020, p. 38).<br><br>Assim, a Demonstração de Resultado Abrangente mostra o desempenho financeiro da empresa, incluindo não apenas o lucro líquido, mas também outros itens abrangentes, como ajustes de conversão de moeda estrangeira e variações em instrumentos de hedge.",
                        
        "Demonstração de Resultado (DRE)": 
        "A Demonstração do Resultado do Exercício (DRE) é um relatório contábil destinado a evidenciar a composição do resultado formado num determinado período de operações da empresa.<br><br>Essa demonstração, observado o Regime de Competência, evidenciará a formação do resultado, mediante confronto entre as receitas e os correspondentes custos e despesas.<br><br>A DRE, portanto, é uma demonstração contábil que evidencia o resultado econômico, isto é, o lucro ou o prejuízo apurado pela empresa no desenvolvimento das suas atividades durante um determinado período que geralmente é igual a um ano (RIBEIRO, 2017a, p. 416).<br><br>Assim, a Demonstração de Resultado apresenta o desempenho financeiro da empresa ao longo de um período, mostrando as receitas, custos, despesas e o lucro líquido.",
                
        "Demonstração de Valor Adicionado (DVA)":        
        "A Demonstração do Valor Adicionado (DVA) é um relatório contábil que evidencia o quanto de riqueza uma empresa produziu, isto é, o quanto ela adicionou de valor aos seus fatores de produção, e o quanto e de que forma essa riqueza foi distribuída (entre empregados, Governo, acionistas, financiadores de capital), bem como a parcela da riqueza não distribuída.<br><br>Desse modo, a DVA tem por finalidade demonstrar a origem da riqueza gerada pela empresa, e como essa riqueza foi distribuída entre os diversos setores que contribuíram, direta ou indiretamente, para a sua geração.<br><br>O valor adicionado que é demonstrado na DVA corresponde à diferença entre o valor da receita de vendas e os custos dos recursos adquiridos de terceiros (RIBEIRO, 2017a, p. 441).<br><br>Assim, a Demonstração de Valor Adicionado evidencia a distribuição da riqueza gerada pela empresa entre os diversos agentes que contribuíram para sua formação, como funcionários, acionistas e governo.",        
        
        "EBITDA": 
        "Uma medida de desempenho financeiro que indica os ganhos operacionais de uma empresa antes das despesas financeiras, impostos e outras deduções.<br><br>Fonte: https://cltlivre.com.br/blog/o-que/ebitda-o-que-e.html",
        
        "ROA": 
        "Uma métrica que mede a eficiência com que uma empresa utiliza seus ativos para gerar lucro.<br><br>Fonte: https://www.empiricus.com.br/explica/roa/",
        
        "ROE": 
        "Uma métrica que avalia a capacidade de uma empresa de gerar retorno para seus acionistas com base no patrimônio líquido.<br><br>Fonte: https://www.suno.com.br/artigos/roe-utilidade/",
        
        "Liquidez Corrente": 
        "Indica a capacidade de uma empresa de pagar suas obrigações de curto prazo utilizando seus ativos de curto prazo.<br><br>Fonte: https://fiibrasil.com/glossario/liquidez-corrente-2/",
        
        "Capital de Giro": "A diferença entre os ativos circulantes (como contas a receber e estoques) e os passivos circulantes (como contas a pagar), indicando a capacidade da empresa de administrar suas operações diárias.<br><br>Fonte: https://sebrae.com.br/sites/PortalSebrae/artigos/artigosFinancas/o-que-e-e-como-funciona-o-capital-de-giro",
        
        "Depreciação": 
        "A alocação sistemática do custo de um ativo tangível ao longo de sua vida útil, refletindo a perda de valor ao longo do tempo.<br><br>Fonte: https://www.empiricus.com.br/explica/depreciacao/",
        
        "Amortização": "O processo de alocação do custo de ativos intangíveis ao longo do tempo, como patentes e marcas registradas.<br><br>Fonte: https://conteudo.cvm.gov.br/export/sites/cvm/menu/regulados/normascontabeis/cpc/CPC_04_R1_rev_12.pdf",
        
        "Margem de Lucro": 
        "A porcentagem de lucro em relação à receita total, indicando a eficiência operacional e a rentabilidade de uma empresa.<br><br>Fonte: https://eprconsultoria.com.br/como-calcular-margem-de-lucro-de-um-produto/",
        
        "Índice de Endividamento": 
        "A relação entre dívida total e patrimônio líquido, refletindo o grau de alavancagem financeira da empresa.<br><br>Fonte: https://blog.leverpro.com.br/post/10-indicadores-de-endividamento-e-alavancagem-financeira",
        
        "Ponto de Equilíbrio": 
        "O nível de vendas em que os custos e despesas totais se igualam à receita, resultando em zero de lucro líquido.<br><br>Fonte: https://www.contabilizei.com.br/contabilidade-online/formula-ponto-de-equilibrio-o-que-e-e-como-calcular-cada-um-dos-tipos/",
                
        "Dividendos": 
        "Pagamentos feitos aos acionistas como parte dos lucros da empresa.<br><br>Fonte: https://blog.toroinvestimentos.com.br/bolsa/dividendos.",
        
        "Ativos Intangíveis": 
        "Recursos não físicos que têm valor econômico, como patentes, marcas registradas e goodwill.<br><br>Fonte: https://www.treasy.com.br/blog/ativo-intangivel/",
        
        "Goodwill": 
        "O valor intangível atribuído a uma empresa como resultado de sua reputação, relacionamentos com clientes, entre outros fatores.<br><br>Fonte: https://blog.americanasmarketplace.com.br/2023/05/15/goodwill",
        
        "Análise Vertical e Horizontal": 
        "Métodos de avaliação da estrutura financeira e desempenho da empresa, comparando informações ao longo do tempo e em relação a um ponto específico.<br><br>Fonte: https://www.suno.com.br/artigos/analise-vertical-e-horizontal/",
        
        "Custos Fixos e Variáveis": 
        "Custos que não mudam com o nível de produção (fixos) versus custos que mudam proporcionalmente com a produção (variáveis).<br><br>Fonte: https://www.portaldecontabilidade.com.br/tematicas/custo-fixo-variavel.htm",
        
        "Comissão de Valores Mobiliários (CVM)":
        "A CVM - Comissão de Valores Mobiliários é uma entidade autárquica em regime especial, vinculada ao Ministério da Fazenda, com personalidade jurídica e patrimônio próprios, dotada de autoridade administrativa independente, ausência de subordinação hierárquica, mandato fixo e estabilidade de seus dirigentes, e autonomia financeira e orçamentária. (Redação dada pela Lei nº 10.411, de 26 de fevereiro de 2002).<br><br>A CVM surgiu com vistas ao desenvolvimento de uma economia fundamentada na livre iniciativa, e tendo por princípio básico defender os interesses do investidor, especialmente o acionista minoritário, e o mercado de valores mobiliários em geral, entendido como aquele em que são negociados títulos emitidos pelas empresas para captar, junto ao público, recursos destinados ao financiamento de suas atividades.<br><br>Ao eleger como objetivo básico defender os investidores, especialmente os acionistas minoritários, a CVM oferece ao mercado as condições de segurança e desenvolvimento capazes de consolidá-lo como instrumento dinâmico e eficaz na formação de poupanças, de capitalização das empresas e de dispersão da renda e da propriedade, através da participação do público de uma forma crescente e democrática e assegurando o acesso do público às informações sobre valores mobiliários negociados e sobre quem os tenha emitido.<br><br>Fonte: https://www.gov.br/cvm/pt-br/acesso-a-informacao-cvm/servidores/estagio/2-materia-cvm-e-o-mercado-de-capitais",

        "DFP – Demonstrações Financeiras Padronizadas":
        "DFP (Demonstrações Financeiras Padronizadas) é um conjunto de informações contábeis que as empresas de capital aberto devem divulgar ao mercado, o que é feito por meio do preenchimento de um formulário, que deve ser enviado à CVM no formato de documento eletrônico.<br><br>Fonte: https://maisretorno.com/portal/termos/d/dfp-demonstracoes-financeiras-padronizadas",
        
        "B3":
        "A B3 é a bolsa de valores do mercado de capitais brasileiro.<br>Apesar de não estar entre as dez maiores do mundo, ela é a maior bolsa de valores da América Latina.<br><br>Fonte: https://www.infomoney.com.br/guias/o-que-e-b3/",
        
    }

    st.write("Selecione um termo abaixo para obter mais informações:")
    termo_selecionado = st.selectbox("Termos", list(glossario.keys()))

    st.write(f"**{termo_selecionado}**")
    
    # https://lightrun.com/answers/streamlit-streamlit-make-it-easier-to-render-newlines-in-markdown
    import re
    phrase = glossario[termo_selecionado]
    phrase = re.sub("<br>"," \n", phrase, flags=re.IGNORECASE)
    st.write(phrase)

# ------------------------------------------------------------------------------

# https://docs.streamlit.io/library/api-reference/text
# https://blog.streamlit.io/introducing-multipage-apps/

st.markdown("# Glossário Financeiro 📖")

# ------------------------------------------------------------------------------
# Página
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

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
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
