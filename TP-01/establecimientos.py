import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
import seaborn as sns

# Cargo la tabla
archivo_distribucion: str = "TP-01/Datos/establecimientos_productivos/distribucion_establecimientos_productivos_sexo.csv"
dist: DataFrame = pd.read_csv(archivo_distribucion, encoding="utf8")

# Analizo la composicion del ID para ver su compatibilidad como clave unica
dist["ID"].isna().sum()  # no hay
dist["ID"].isnull().sum()  # no hay
dist["ID"].describe()  # vemos que son todas unicas
