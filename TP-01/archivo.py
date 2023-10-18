'''
Autores:    
    gantes, augusto
    
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hacemos copia de los datasets originales para luego afectarlos
archivo_padron = 'TP-01/TablasOriginales/padron-de-operadores-organicos-certificados.csv'
padron = pd.read_csv(archivo_padron, encoding='latin1').copy(deep=True)
archivo_establecimientos = 'TP-01/TablasOriginales/distribucion_establecimientos_productivos_sexo.csv'
establecimientos = pd.read_csv(archivo_establecimientos).copy(deep=True)
archivo_clae = 'TP-01/TablasOriginales/clae_agg.csv'
clae = pd.read_csv(archivo_clae).copy(deep=True) 
archivo_localidad = 'TP-01/TablasOriginales/localidad_bahra.csv'
localidad = pd.read_csv(archivo_localidad).copy(deep=True) 


# Tablas Limpias
###############################################################################

# TODO crear tablas vacias para luego importar datos


###############################################################################

# Limpieza de datos
###############################################################################

# Padron

# Empezamos por sacar informacion irrelevante para el objetivo
padron.drop(columns=["pais_id",
                     "pais",
                     "Certificadora_id",
                     "certificadora_deno",
                     "categoria_id",
                     "categoria_desc"],
            inplace=True)


###############################################################################


