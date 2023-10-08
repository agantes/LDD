import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

archivo: str = "TP-01/Datos/fuentes_secundarias/clae_agg.csv"
df: DataFrame = pd.read_csv(archivo, encoding="utf8")

# Comprobamos nans y nulls
df.isna().any(axis=0)
df.isnull().any(axis=0)

# Vemos que tanto clae3, clae2 y letra tienen repeticion contsantemente
# Creo que clae6 sirve como identificador, lo que nos podria llevar a 2FN
df["clae6"].describe()
df["clae6"].value_counts().max()  # vemos que son únicos
