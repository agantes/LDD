import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
import seaborn as sns

# TODO Debuggear juntar_dep

"""
Esta fuente de datos tiene la utilidad de permitirnos asociar el Padrón de 
Operadores con los datos de departamento
"""

def normalizar_str(df: DataFrame, col: str) -> None:
    """
    Esta función devuelve una serie de reemplazo del DataFrame por una
    que contiene minusculas y sin tildes. 
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
    
    return serie_res

def juntar_dep(padron: pd.DataFrame, localidad: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    padron : pd.DataFrame
        El primer argumento debera ser el DataFrame correspondiente al Padron
        de productores organicos el cual ya debera estar en minusculas y sin
        tildes.
    localidad : pd.DataFrame
        El segundo argumento debera ser el DataFrame correspondiente a 
        Localidad el cual ya debera estar en minusculas y sin tildes.
    Returns
    Un DaataFrame en minuscula y sin tildes con una unica columna de provincia
    donde se juntan los departamentos de un DataFrame con el otro teniendo en
    cuenta la prioridad de encontrar un coincidente para cada departamento de 
    el DataFrame correspondiente a el Padron de productores organicos.
    """
    
    def coincidentes(s1: str, s2: str) -> int:
        """
        Con dos strings genera un puntaje de coincidencia para determinar 
        similitud por igualdad en cantidad de palabras separadas por espacios.
        Se remplazan las comas por espacios vacios para luego llevar a cabo la
        comparacion.
        """
        
        # Sacamos comas y separamos por espacio
        s1_mod, s2_mod = s1.replace(",", "").split(), s2.replace(",", "").split()
        
        # Iteramos para conseguir match segun que tan parecida es la segunda a 
        # la primera
        res: int = 0
        for ele1 in s1_mod:
            for ele2 in s2_mod:
                if ele1 == ele2:
                    res += 1
                else: continue
        return res
    
    # Generamos la estructura final
    res_dicc: dict = {
        "provincia": [], 
        "departamento_padron" : [],
        "departamento_localidad" : []
        }
    for provincia in padron["provincia"].unique():
        # Hacemos un filtro de provincia
        padron_provincia = df[df["provincia"] == provincia]
        localidad_provincia = localidad[localidad["nombre_provincia"] == provincia]
        # Ahora vemos que entrada es la más parecida según la entrada de padron
        # y las añadimos juntas
        for dep_padron in padron_provincia["departamento"]:
            max_puntaje: int = 0
            max_match: str = ""
            for dep_localidad in localidad_provincia["nombre_departamento"]:
                puntaje: int = coincidentes(dep_padron, dep_localidad) 
                if puntaje >= max_puntaje:
                    max_puntaje = puntaje
                    max_match = dep_localidad
                puntaje = 0
            res_dicc["provincia"].append(provincia)
            res_dicc["departamento_padron"].append(dep_padron)
            res_dicc["departamento_localidad"].append(dep_localidad)

    return pd.DataFrame(res_dicc)


archivo_loc: str = "TP-01/Datos/fuentes_secundarias/localidad_bahra.csv"
loc: DataFrame = pd.read_csv(archivo_loc, encoding="utf8").copy()
archivo: str = "TP-01/Datos/padron_operadores/padron-de-operadores-organicos-certificados.csv"
df: DataFrame = pd.read_csv(archivo, encoding="latin-1").copy()
archivo_distribucion: str = "TP-01/Datos/establecimientos_productivos/distribucion_establecimientos_productivos_sexo.csv"
dist: DataFrame = pd.read_csv(archivo_distribucion, encoding="utf8").copy()

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

# Dropeo aglomerado y tipo de asentamiento, no hay info cruzable con padron
# De padron solo uso provincia y departamento
loc.drop(columns=["nombre_aglomerado",
                  "tipo_asentamiento",
                  "nombre_geografico"], inplace=True)
df.drop(columns=list(set(df.columns) - {"provincia", "departamento"}),
        inplace=True)
df_head = df.head()
df_describe = df.describe()
df_cols = df.columns
loc_head = loc.head()
loc_describe = loc.describe()
loc_cols = loc.columns

# Saco tildes y minusculas de loc y departamento de padron
for col in loc_cols:
    loc[col] = normalizar_str(loc, col)
for col in df_cols:
    df[col] = normalizar_str(df, col)
    
# ATENCION: Hay un caso en particular, el de Ciudad de Buenos Aires, tiene
# diferente nombre la provincia y en localidad engloba todas las comunas, es 
# decir, no esta en 1FN. 
# Lo voy a sacar y les hacemos un tratamiento aparte con reemplazos
# Me centro en la anterior tarea por ahora, los dropeo de momento, son 40 datos
loc_bsas = loc["nombre_provincia"] == "ciudad de buenos aires"
loc = loc.drop(loc[loc_bsas].index)
df_bsas = df["provincia"] == "ciudad autonoma buenos aires"
df = df.drop(df[df_bsas].index)

# Ahora aplico la función que genera las coincidencias
dep_totales = juntar_dep(df, loc)
