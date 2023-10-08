import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
import seaborn as sns

# cargamos el archivo, el encondong es correspondiente a windows
archivo: str = "TP-01/Datos/padron_operadores/padron-de-operadores-organicos-certificados.csv"
df: DataFrame = pd.read_csv(archivo, encoding="latin-1")

# veo la composicion junto a descriptores basicos
df_des = df.describe()
df_head = df.head()
df_cols = df.columns

# veo que muy posiblemente no hay necesidad de pais_id ni pais, todos iguales
# tampoco veo utilidad en certificadoras para el objetivo
df = df.drop(columns=["pais_id",
                      "pais", 
                      "Certificadora_id",
                      "certificadora_deno",
                      "establecimiento"])

# analizo si hay nulls o nans
df.isna().any(axis=0)
df.isnull().any(axis=0)

# Podemos ver que hay nulls y nans en rubro y productos
# Analizo que tan grave es el problema para ambas columnas
def check_val(df: DataFrame, col: str) -> float:
    """
    Dado un pandas.DataFrame y una columna válida, la función devuelve
    el porcentaje de nulls y nans que representa para el total de la columna.    
    """
    nan: int = df[col].isna().sum().sum()
    null: int = df[col].isnull().sum().sum()
    invalid: int = nan + null
    total: int = len(df)
    return round(100*(invalid/total), 8)

rubro_inval: float = check_val(df, "rubro")
productos_inval: float = check_val(df, "productos")

# No llegan a representar un 1% de la muestra, los elimino
df = df.dropna()
df.isna().any(axis=0)
df.isnull().any(axis=0)

# TODO Calidad de Datos
# Vemos valores unicos en cols de categoria que suponemos útiles
# para ver si hay algo mas que nans y nulls porque vi que hay

# Analizo la col departamento
# Faltan tildes, problemas de compatibilidad con fuente de localidad
dep_vals = df["departamento"].value_counts()
sns.kdeplot(data=dep_vals)
plt.grid()
plt.show()
plt.close()

# Se observa un cumulo de valores y luego ciertos outliers con gran numero
# No se observa ningun valor invalido tipo null/nan o clasificacion propia
# de los creadores de la base de datos
