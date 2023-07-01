# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("./")

# ------------------------------------------------------------------------------

# https://stackoverflow.com/questions/52645710/a-list-of-downloaded-files-names-in-google-colabarotary
# https://www.geeksforgeeks.org/python-filter-list-of-strings-based-on-the-substring-list/

def Filter(string, substr):

    # Essa função filtra uma lista de strings e retorna apenas aquelas que contêm alguma string especificada em outra lista de strings (substr).
    # Ela percorre cada string em string e verifica se alguma string em substr está contida na string atual.
    # Se sim, a string atual é adicionada à lista de resultados.

    return [str for str in string if any(sub in str for sub in substr)]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------ 
