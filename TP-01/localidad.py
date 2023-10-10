import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
import seaborn as sns

"""
Esta fuente de datos tiene la utilidad de permitirnos asociar el Padrón de 
Operadores con los datos de departamento
"""

archivo: str = "TP-01/Datos/fuentes_secundarias/localidad_bahra.csv"
loc: DataFrame = pd.read_csv(archivo, encoding="utf8").copy()

# Veo descriptores básicos
loc_head = loc.head()
loc_describe = loc.describe()
loc_cols = loc.columns

# A primera vista, veo que columnas no son de utilidad para el objetivo de este
# archivo 
# Veo que las propias de locaclizacion exacta son de poca utilidad para el 
# objetivo del trabajo
# Tambien dropeo identificadores y codigos
loc.drop(loc.columns[:2], axis=1, inplace=True)  # ids
loc.drop(loc.columns[8:], axis=1, inplace=True)  # datos geom 
loc.drop(columns=["codigo_asentamiento", 
                 "codigo_aglomerado", 
                 "codigo_indec_departamento"],
        inplace=True)  # codigos
loc_cols = loc.columns

# Analizo si hay nulls o nans
loc.isna().any(axis=0)
loc.isnull().any(axis=0)

# La unica con este problema es nombre_aglomerado
loc["nombre_aglomerado"].isna().sum()
loc["nombre_aglomerado"].isnull().sum()

# Son 6 en total para cada, vemos la tabla de null
agl_na = loc[loc["nombre_aglomerado"].isna() == True]
agl_null = loc[loc["nombre_aglomerado"].isnull() == True]

# Si vemos utilidad en Aglomerado, eliminar nulls, no son representativos
# loc.dropna(inplace=True)

# Analizo nombre geografico
geo_vals = loc["nombre_geografico"].value_counts()
loc["nombre_geografico"].describe()
sns.kdeplot(data=geo_vals)
plt.title("Count Geo")
plt.grid()
plt.show()
plt.close()

# Analizo nombre departamento
dep_vals = loc["nombre_departamento"].value_counts()
loc["nombre_departamento"].describe()
sns.histplot(data=dep_vals)
plt.title("Count Departamentos")
plt.grid()
plt.show()
plt.close()

# Analizo nombre provincia
prov_vals = loc["nombre_provincia"].value_counts()
loc["nombre_provincia"].describe()
sns.histplot(data=prov_vals)
plt.title("Count Provincias")
plt.grid()
plt.show()
plt.close()

# Analizo nombre aglomerado
# Hay nans, no los removi
agl_vals = loc["nombre_aglomerado"].value_counts()
loc["nombre_aglomerado"].describe()
sns.kdeplot(data=agl_vals)
plt.title("Count Aglomerado")
plt.grid()
plt.show()
plt.close()
