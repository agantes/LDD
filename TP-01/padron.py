import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

"""
Este archivo de padron nos da una idea de las actividades de producción 
orgánica.
"""

# TODO Calidad de Datos
# Vemos valores unicos en cols de categoria que suponemos útiles
# para ver si hay algo mas que nans y nulls porque vi que hay

# Funciones
def diferencia_cjto(cjto1, cjto2) -> int:
  """
  Esta función toma dos cjtos y cuenta la diferencia devolviendola en
  forma de integer.
  """
  union = cjto1 | cjto2
  num_dif = len(union)
  return num_dif

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

def normalizar_str(df: DataFrame, col: str) -> None:
    """
    Esta función reemplaza la una columna inplace del DataFrame por una
    que contiene minusculas y sin minusculas. 
    """
    # Definimos lista con vocales sin y con tilde
    vocales: list[str] = ["a", "e", "i", "o", "u"]
    vocales_con_tilde: list[str] = ["á", "é", "í", "ó", "ú"]
    
    def idx(x: str, l: list) -> int:
        j: int = 0
        res: int = 0
        while j < len(l):
            if l[j] == x:
                res = j
                break
            j += 1
        return res
    
    # Conseguimos la serie de la col que queremos afectar
    serie_str: list[str] = list(df[col])
    serie_res: list[str] = list()
    for ele in serie_str:
        ele_mod: str = ""
        for i in ele:
            if i in vocales_con_tilde:
                ele_mod += vocales[idx(i,vocales_con_tilde)]
            else:
                ele_mod += i
        serie_res.append(ele_mod.lower())
    
    df[col] = serie_res


# Cargamos el archivo, el encondong es correspondiente a windows
# Agrego copy hasta la confirmacion de los cambios
# Hay muchas instancias con el uso de inplace, y no quiero updatear el df en git
archivo: str = "TP-01/Datos/padron_operadores/padron-de-operadores-organicos-certificados.csv"
df: DataFrame = pd.read_csv(archivo, encoding="latin-1").copy()
archivo_loc: str = "TP-01/Datos/fuentes_secundarias/localidad_bahra.csv"
loc: DataFrame = pd.read_csv(archivo_loc, encoding="utf8").copy()


# veo la composicion junto a descriptores basicos
df_des = df.describe()
df_head = df.head()
df_cols = df.columns

# veo que muy posiblemente no hay necesidad de pais_id ni pais, todos iguales
# tampoco veo utilidad en certificadoras para el objetivo
# Dejo datos utiles para anlisis
df.drop(columns=["pais_id",
                 "pais",
                 "Certificadora_id",
                 "certificadora_deno",
                 "establecimiento",
                 "provincia_id",
                 "categoria_id",
                 "categoria_desc",
                 "razón social"],
        inplace=True)

# analizo si hay nulls o nans
df.isna().any(axis=0)
df.isnull().any(axis=0)

# Podemos ver que hay nulls y nans en rubro y productos
# Analizo que tan grave es el problema para ambas columnas
rubro_inval: float = check_val(df, "rubro")
productos_inval: float = check_val(df, "productos")

# No llegan a representar un 1% de la muestra, los elimino
df = df.dropna()
df.isna().any(axis=0)
df.isnull().any(axis=0)

# Analizo la col departamento
# Faltan tildes, problemas de compatibilidad con fuente de localidad
dep_vals = df["departamento"].value_counts()
sns.kdeplot(data=dep_vals)
df["departamento"].describe()
plt.grid()
plt.show()
plt.close()

# Se observa un cumulo de valores y luego ciertos outliers con gran numero
# No se observa ningun valor invalido tipo null/nan o clasificacion propia
# de los creadores de la base de datos

# Analizo la col rubro que tiene datos del tipo 'SIN DEFINIR'
rubro_sd = df[df["rubro"] == "SIN DEFINIR"] 
rubro_inval_sd = 100*(len(rubro_sd)/len(df))  # casi un 8% del df

# Los filtro y veo que nos queda a disposición
df.drop(rubro_sd.index, inplace=True)
rubros = df["rubro"].unique()

# Vefifico que coincidencias hay entre los departamento de la fuente secundaria
# localidad y esta fuente primaria. Primero trato de pasar todo a minusculas.
normalizar_str(loc, "nombre_departamento")  # normalizo dep de localidad
normalizar_str(df, "departamento")  # normalizo dep de padron

# Vemos si realmente concuerdan los valores unicos de uno y otro
dep_padron = set(df["departamento"].unique())
dep_localidad = set(loc["nombre_departamento"].unique())
dep_localidad == dep_padron  # False

diferencia_cjto(dep_localidad, dep_padron)

# Vemos que hay 623 departamentos con el nomrbre diferente
# Armo un algoritmo para ver similitud
# Necesito encontrar el match para poder ver bien la correspondencia
# de cada uno de los 356 departamentos unicos de padron
