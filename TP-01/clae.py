import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

archivo: str = "TP-01/Datos/fuentes_secundarias/clae_agg.csv"
dicc: DataFrame = pd.read_csv(archivo, encoding="utf8").copy()

# Comprobamos nans y nulls
dicc.isna().any(axis=0)
dicc.isnull().any(axis=0)

# Vemos que tanto clae3, clae2 y letra tienen repeticion contsantemente
# Creo que clae6 sirve como identificador, lo que nos podria llevar a 2FN
dicc["clae6"].describe()
dicc["clae6"].value_counts().max()  # vemos que son únicos

# Analizo los valores únicos por parte del CLAE2
dicc["letra_desc"].unique()
dicc_letra = dicc[(dicc["letra"] == "A") | (dicc["letra"] == "H")]

# Lo que se ve claramente es que hay un uso repetitivo y sistemático sobre
# los verbos para iniciar las actividades y "Servicio"

# Se puede apreciar que para vincular los CLAE más al ambito organico
# podemos seguir filtrando selectivamente segun el CLAE3
# Esta es una lista de los posibles campos organicos
campos_organicos = [
    " Cultivos temporales",
    " Cultivos perennes",
    " Producción de semillas y de otras formas de propagación de cultivos agrícolas",
    " Silvicultura",
    " Extracción de productos forestales"
    ]


# limpio la tabla de claes con lo que nos importa a nosotros
archivo_clae = 'TP-01/TablasOriginales/clae_agg.csv'
clae = pd.read_csv(archivo_clae).copy(deep=True) 

clae.drop(columns=["clae6",
                 "clae6_desc",
                 "clae3",
                 "clae3_desc"],
        inplace=True)

clae_cols = clae.columns
#print(clae_cols) verifico que se hayan eliminado las columnas 
dicc_sin_duplicados = dicc.drop_duplicates(subset=['clae2', 'clae2_desc']) #creo la nueva tabla
