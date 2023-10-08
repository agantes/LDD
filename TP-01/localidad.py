import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
import seaborn as sns

"""
Esta fuente de datos tiene la utilidad de permitirnos asociar el Padrón de 
Operadores con los datos de departamento
"""

archivo: str = "TP-01/Datos/fuentes_secundarias/localidad_bahra.csv"
df: DataFrame = pd.read_csv(archivo, encoding="utf8")

# Veo descriptores básicos
df_head = df.head()
df_describe = df.describe()
df_cols = df.columns

# A primera vista, veo que columnas no son de utilidad para el objetivo de este
# archivo 
# Veo que las propias de locaclizacion exacta son de poca utilidad para el 
# objetivo del trabajo
# Tambien dropeo identificadores y codigos
df.drop(df.columns[:2], axis=1, inplace=True)  # ids
df.drop(df.columns[8:], axis=1, inplace=True)  # datos geom 
df.drop(columns=["codigo_asentamiento", 
                 "codigo_aglomerado", 
                 "codigo_indec_departamento"],
        inplace=True)  # codigos
df_cols = df.columns

# Analizo si hay nulls o nans
df.isna().any(axis=0)
df.isnull().any(axis=0)

# La unica con este problema es nombre_aglomerado
df["nombre_aglomerado"].isna().sum()
df["nombre_aglomerado"].isnull().sum()

# Son 6 en total para cada, vemos la tabla de null
agl_na = df[df["nombre_aglomerado"].isna() == True]
agl_null = df[df["nombre_aglomerado"].isnull() == True]

# Si vemos utilidad en Aglomerado, eliminar nulls, no son representativos
# df.dropna(inplace=True)

# Analizo nombre geografico
geo_vals = df["nombre_geografico"].value_counts()
df["nombre_geografico"].describe()
sns.kdeplot(data=geo_vals)
plt.title("Count Geo")
plt.grid()
plt.show()
plt.close()

# Analizo nombre departamento
dep_vals = df["nombre_departamento"].value_counts()
df["nombre_departamento"].describe()
sns.histplot(data=dep_vals)
plt.title("Count Departamentos")
plt.grid()
plt.show()
plt.close()

# Analizo nombre provincia
prov_vals = df["nombre_provincia"].value_counts()
df["nombre_provincia"].describe()
sns.histplot(data=prov_vals)
plt.title("Count Provincias")
plt.grid()
plt.show()
plt.close()

# Analizo nombre aglomerado
# Hay nans, no los removi
agl_vals = df["nombre_aglomerado"].value_counts()
df["nombre_aglomerado"].describe()
sns.kdeplot(data=agl_vals)
plt.title("Count Aglomerado")
plt.grid()
plt.show()
plt.close()
