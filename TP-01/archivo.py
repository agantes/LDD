'''
Autores:    
    gantes, augusto
    
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from inline_sql import sql

def check_val(df: pd.DataFrame, col: str) -> float:
    '''
    Dado un pandas.DataFrame y una columna válida, la función devuelve
    el porcentaje de nulls y nans que representa para el total de la columna.    
    '''
    nan: int = df[col].isna().sum().sum()
    null: int = df[col].isnull().sum().sum()
    invalid: int = nan + null
    total: int = len(df)
    return round(100*(invalid/total), 8)


def normalizar_str(df: pd.DataFrame, col: str) -> None:
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
    
    return serie_res
    

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
padron.drop(columns=['pais_id',
                     'pais',
                     'Certificadora_id',
                     'certificadora_deno',
                     'categoria_id',
                     'categoria_desc'],
            inplace=True)

# Vemos nulls y nans
padron.isna().any(axis=0)
padron.isnull().any(axis=0)

# Rubro y productos tienen nans/nulls
# Evaluamos que tan grave es el problema
check_val(padron, 'rubro')
check_val(padron, 'productos')

# Son pocos, ambos menos del 1%, los eliminamos ya que dificultan la inferencia
padron = padron.dropna()

# Localidad es una columna inutilizada, se ve que hay un valor propio del df
# para aclarar que no esta definido el valor
# Vemos valores sin definir
indefinido = 0
for localidad in padron['localidad']:
    if localidad == 'INDEFINIDO' or localidad == 'INDEFINIDA':
        indefinido += 1
localidades_sin_definir = 100 * (indefinido / len(padron['localidad']))
localidades_sin_definir  # porcentaje sin definir

# Sacamos localidad, hay un 96% del df que no sirve para nada
padron = padron.drop('localidad', axis=1)

# Reemplazamos los rubros sin definir por alguna categoria relacionada con 
# letra de clae segun los productos
# Cargamos una tabla auxiliar hecha manualmente
archivo_imputacion_sin_definir = 'TP-01/TablasLimpias/rubro_clae_imputacion_sin_definir.csv'
imputacion_rubro_sin_definir = pd.read_csv(archivo_imputacion_sin_definir)

# Hacemos un merge
df_merge_imputacion = pd.merge(padron, imputacion_rubro_sin_definir, 
                               left_on='productos',
                               right_on='prodcutos',
                               how='left')
df_merge_imputacion['valor_definitivo'] = df_merge_imputacion[
    'rubro_y'].fillna(padron['rubro'])
df_res_imputacion = df_merge_imputacion[['valor_definitivo']]

# Asigno la data imputada a el df 
padron['rubro'] = df_res_imputacion['valor_definitivo']

# Por alguna razon que desconozco, los shapes no concuerdan
# Hay un valor que por el merge queda desacoplado, dando como resultado
# un nan
# Lo limpio pero no hay justificacion clara de lo sucedido, probablemente
# hubo un error en la imputacion manual
# Hay una perdida de 5 operadores
padron = padron.dropna()

# Lo que sigue es hacer una reparacion apropiada de la columna de departamentos
# Se observa que hay poca data que haga referencia a un departamento 
# Primero, sacamos tildes y pasamos a minuscula
padron['departamento'] = normalizar_str(padron, 'departamento')
padron['provincia'] = normalizar_str(padron, 'provincia')
localidad['nombre_departamento'] = normalizar_str(
    localidad, 'nombre_departamento')
localidad['nombre_provincia'] = normalizar_str(
    localidad, 'nombre_provincia')
departamentos_con_coincidencia = len(
    set(padron['departamento'].unique()) & set(localidad['nombre_departamento'].unique()))
departamentos_unicos = len(padron['departamento'].unique())
porcentaje_departamentos_registrables = 100 * (
    departamentos_con_coincidencia / departamentos_unicos)
porcentaje_departamentos_registrables 

# Observamos que hay un 46% de los departamentos que, por alguna razon, no 
# tienen coincidencia alguna con la base de datos de BAHRA
# Podemos incluso ver si los que no aparecen en departamento estan en alguna
# otra columna de la tabla de localidad

# Se realizo una inputacion a mano por ciertos problemas observados
# Cargamos la tabla auxiliar que nos permite realizar una imputacion 
# similar a la de rubro
archivo_imputacion_departamento = 'TP-01/TablasLimpias/departamentos_imputacion.csv'
imputacion_departamento = pd.read_csv(archivo_imputacion_departamento)

# Genero un merge con la data que esta en departamento
df_merge_imputacion = pd.merge(padron, imputacion_departamento, 
                               on=['provincia', 'departamento'],
                               how='left')
df_merge_imputacion['valor_definitivo'] = df_merge_imputacion[
    'departamento_bahra'].fillna(localidad['nombre_departamento'])
df_res_imputacion = df_merge_imputacion[['valor_definitivo']]

# Asignamos el valor final
padron['departamento'] = df_res_imputacion['valor_definitivo']

# Si corremos la metrica anterior que verifica si tenemos los departamentos
# de padron pertenecientes a la base de localidad ahora da 100%, la imputacion
# fue exitosa o eso parece

###############################################################################


