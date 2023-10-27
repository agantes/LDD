"""
Materia     : Laboratorio de datos - FCEyN - UBA
Autores     : Augusto Gantes, Martin Belmes y Matias D'Andrea
Detalle     : 
    Este archivo contiene el desarrollo de todas las actividades dispuestas 
    en el TP-02 de manera tal que se ejecute el código por secciones. Se 
    comentaran los hallazgos a lo largo del codigo. Decidir que hacer con 
    funciones auxiliares, si es que usamos.
    
    Importante: se asume que el proyecto de Spyder esta situado en la carpeta
    en la que se encuentran todos los archivos. NO entregar dataset.
    
Creacion    : 25/10/2023
Modificacion: 26/10/2023
"""

# %% Importacion de librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# %% Carga de Datos
df = pd.read_csv('fashion-mnist.csv', encoding='utf-8')

# %% Funciones


def std_pixeles(df: pd.DataFrame, label=True) -> pd.DataFrame:
    '''
    Transforma el df en otro con los pixeles en una columna 'posicion' y 
    la desviacion estandar en una columna 'std'. Dependiendo del parametro
    'label' verificamos si es que el df viene con una columna con las etiquetas
    y consecuentemente se la retiramos. Si esta la necesidad de filtrar por 
    etiqueta, hacerlo previamente.
    '''
    if label:
        posiciones = list(set(df.columns) - {'label'})
    else:
        posiciones = df.columns
    std_por_posicion = {'posicion': [], 'std': []}
    for posicion in posiciones:
        numero_posicion_str: list = re.findall(r'\d+', posicion)
        numero_posicion = int(numero_posicion_str[0])
        std_por_posicion['posicion'].append(numero_posicion)
        std_por_posicion['std'].append(df[posicion].std())

    return pd.DataFrame(std_por_posicion)


def plot_std_pixeles(df: pd.DataFrame, title: str) -> None:
    '''
    Genera un plot de distribución por kernel junto a un scatterplot para
    graficar la data correspondiente a df resultantes de std_pixeles. Aclarar
    titulo como parametro.
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=df, x='posicion', y='std',
                color='black', alpha=.5)
    sns.scatterplot(data=df, x='posicion', y='std')
    ax.set_xlabel('Pixel', fontsize=18)
    ax.set_ylabel('Desviación Estándar', fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid()
    plt.show()
    plt.close()


def map_std(df: pd.DataFrame, cota: float) -> np.array:
    '''
    Esta funcion nos permite generar una matriz de 28 x 28 que contiene todos
    los pixeles y resalta si estan por debajo de cierta desviación estándar.
    El df de entrada tiene que ser aquel resultante de std_pixeles.
    '''
    # Ordenamos las posiciones para poder romper aporpiadamente
    # el dataframe
    df = df.sort_values('posicion')
    stds = list(df['std'])

    # Voy generando la matriz como lista de listas
    res = list()
    current_row = list()
    i: int = 1
    while i <= 784:
        if stds[i - 1] <= cota:
            current_row.append(1)
        else:
            current_row.append(0)

        # El 27 viene por el index 0, esto implica que terminamos la fila
        if (i % 28 == 0) and (i != 0):
            res.append(current_row)  # agrego a la matriz
            current_row = []  # reseteo el acumulador
        i += 1

    return np.array(res)


def plot_map_std(mat: np.array, title: str) -> None:
    '''
    Plotea la data conseguida por map_std en forma de imagen de 28 x 28
    con matplotlib. 
    '''
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.imshow(mat, cmap='Greys', origin='upper')
    plt.title(title)
    plt.show()
    plt.close()


# %% Analisis de los Datos

# Para empezar analizamos la cantidad de datos y descriptores básicos
df.head()
df.describe()  # el maximo y cuartiles varian bastante entre posiciones
df.info()

# %%% Desviacion estandar (std) por Pixel

# Segun lo que vemos de lo variable que es la data segun posicion de pixel
# nos preguntamos si es que hay posiciones que siempre sean de una misma
# intensidad en la escala de grises, i.e., puede que no haya std.
std_por_posicion = std_pixeles(df)  # Modifico el df para analizar std p/pixel

# Vemos como se distribuyen las disperciones
plot_std_pixeles(std_por_posicion, 'Desviación Estándar por Píxel')

# Se observan muchas variables con poca variacion, incluso sin haber separado
# por categoria. Tambien podemos ver que hay un cumulo de valores en la parte
# inferior izquierda que contiene valores con std por debajo de 20.

# Genero una matrix de aquellos pixeles que tienen la desviacion estandar por
# debajo de 20
cota = 20
mat_std = map_std(std_por_posicion, cota)
plot_map_std(mat_std, 'Pixeles con Desviación Estándar menor a ' + str(cota))

# Descarto variables que se usaron para graficar y computar
del std_por_posicion, cota, mat_std

# %%% Separacion por etiqueta
# Vemos el recuento de clasificaciones
df['label'].value_counts()  # es uniforme, 6000 entradas para las 10 etiquetas

# Vemos si realmente el dataset contiene informacion diferente para cada label
# diferente. Graficamos
etiquetas = df['label'].unique()
etiquetas.sort()

# Vemos que sucede con cada prenda
for etiqueta in etiquetas:
    # Filtramos por etiqueta
    df_etiqueta = df[df['label'] == etiqueta]
    df_etiqueta = df_etiqueta.drop('label', axis=1)

    # Calculamos desviación y graficamos
    df_etiqueta_std_pixel = std_pixeles(df_etiqueta)
    plot_std_pixeles(df_etiqueta_std_pixel,
                     'Desviación Estándar por Píxel Etiqueta ' + 
                     str(etiqueta))

    # Vemos los pixeles que "no varian mucho" (en comparacion a otros)
    cota = 20
    mat_etiqueta = map_std(df_etiqueta_std_pixel, cota)
    plot_map_std(mat_etiqueta,
                 'Pixeles con Desviación Estándar baja Etiqueta ' + 
                 str(etiqueta))

# Se puede apreciar que según distintos tipos de prenda cambia bastante la
# forma de los gráficos, hasta se puede distinguir si son calzados o no según
# la forma de los datos, lo que indica que hay cierto grdo de acierto en lo que
# estamos analizando
