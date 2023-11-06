"""
Materia     : Laboratorio de datos - FCEyN - UBA
Autores     : Augusto Gantes, Martin Belmes y Matias D'Andrea
Detalle     : 
    Este archivo contiene solamente funciones relacionadas con la exploración de 
    los datos.    
Creacion    : 27/10/2023
Modificacion: 3/11/2023
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

'''
Este archivo contiene solamente funciones relacionadas con la exploración de 
los datos.
'''

def std_pixeles(df: pd.DataFrame, label: bool = True) -> pd.DataFrame:
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

def mean_pixeles(df: pd.DataFrame, label:bool = True) -> pd.DataFrame:
    '''
    Transforma el df en otro con los pixeles en una columna 'posicion' y 
    la desviacion estandar en una columna 'mean'. Dependiendo del parametro
    'label' verificamos si es que el df viene con una columna con las etiquetas
    y consecuentemente se la retiramos. Si esta la necesidad de filtrar por 
    etiqueta, hacerlo previamente.
    '''
    if label:
        posiciones = list(set(df.columns) - {'label'})
    else:
        posiciones = df.columns
    mean_por_posicion = {'posicion': [], 'mean': []}
    for posicion in posiciones:
        numero_posicion_str: list = re.findall(r'\d+', posicion)
        numero_posicion = int(numero_posicion_str[0])
        mean_por_posicion['posicion'].append(numero_posicion)
        mean_por_posicion['mean'].append(df[posicion].mean())

    return pd.DataFrame(mean_por_posicion)

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

def map_metrica(df: pd.DataFrame, metrica: str) -> np.array:
    '''
    Esta funcion nos permite generar una matriz de 28 x 28 que contiene todos
    los pixeles y le adjunta a cada pixel alguna metrica. El df de entrada 
    tiene que ser aquel resultante de std_pixeles o mean_pixeles
    '''
    # Ordenamos las posiciones para poder romper aporpiadamente
    # el dataframe
    df = df.sort_values('posicion')
    metrica = list(df[metrica])

    # Voy generando la matriz como lista de listas
    res = list()
    current_row = list()
    i: int = 1
    while i <= 784:
        # Agregamos la metrica al pixel corriente
        current_row.append(metrica[i - 1])

        # El 27 viene por el index 0, esto implica que terminamos la fila
        if (i % 28 == 0) and (i != 0):
            res.append(current_row)  # agrego a la matriz
            current_row = []  # reseteo el acumulador
        i += 1

    return np.array(res)


def plot_map(mat: np.array, title: str) -> None:
    '''
    Plotea la data conseguida por map_metrica en forma de imagen de 28 x 28
    con matplotlib. 
    '''
    fig, ax = plt.subplots(figsize=(4,4))
    plt.imshow(mat, cmap='inferno', origin='upper')
    plt.title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(shrink=0.8)
    plt.show()
    plt.close()

def plot_maps(mat1: np.array, title1: str,
              mat2: np.array, title2: str) -> None:
    '''
    Genera un plot en paralelo de dos np.arrays basandose en las otras
    funciones. El argumento mat1 hace referencia a la matriz correspondiente
    a la desviación estándar producida por map_metrica, mientras que mat2 esta
    vinculada a la misma función pero sobre el promedio.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im1 = axes[0].imshow(mat1, cmap='inferno', origin='upper')
    axes[0].set_title(title1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, shrink=.7, ax=axes[0], 
                 ticks=np.arange(np.min(mat2), np.max(mat2), 10.0))
    im2 = axes[1].imshow(mat2, cmap='inferno', origin='upper')
    axes[1].set_title(title2)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, shrink=.7, ax=axes[1],
                 ticks=np.arange(np.min(mat2), np.max(mat2), 20.0))
    plt.show()
    plt.close()

def plot_promedios_clases(df: pd.DataFrame) -> None:
    '''
    Esta función genera graficos de promedio entre todas las etiquetas
    en una cuadrilla de 4 x 3. No incluye colorbar y los gráficos seran 
    realizados con la media.
    '''
    # Obtengo las etiquetas
    etiquetas = df['label'].unique()
    etiquetas.sort()
    
    # Genero la figura y adjunto las etiquetas con los ejes
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    ejes = []
    for row in axes:
        for ele in row:
            ejes.append(ele)
    ejes = ejes[:10]  # seleccionamos hasta el 8 
    etiqueta_eje = zip(etiquetas, ejes)
    
    # Itero por etiqueta
    for etiqueta, ax in etiqueta_eje:
        
        # Filtramos por etiqueta
        df_etiqueta = df[df['label'] == etiqueta]
        df_etiqueta = df_etiqueta.drop('label', axis=1)
        
        # Caculo data necesaria para la media
        df_etiqueta_mean_pixel = mean_pixeles(df_etiqueta, label=False)
        mat_mean_etiqueta = map_metrica(df_etiqueta_mean_pixel, 'mean')
        
        # Ploteo la media en la figura
        ax.matshow(mat_mean_etiqueta, cmap='inferno', origin='upper')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Etiqueta {etiqueta}', fontsize=12)
    
    # Eliminamos plots vacios
    fig.delaxes(axes[2][2])
    fig.delaxes(axes[2][3])
    fig.suptitle('Promedios por Clase', fontsize=14)
    plt.show()
    plt.close()
    
def recuperar_posciciones(cota_inferior_vertical: int,
                          cota_superior_vertical: int,
                          cota_inferior_horizontal: int,
                          cota_superior_horizontal: int) -> list:
    """
    Esta función tiene como objetivo recuperar píxeles según ciertas cotas
    que tengan como referencia a la imagen cudrada de 28 x 28. El reultado
    retorna una lista de listas, donde cada lista contiene tres pixeles que 
    corresponden a los pixeles acotados ordenados de tal manera que son 
    consecutivos horizontalmente. 
    Esta función esta dedicada a la sección auxiliar que nos permite ver
    que pixeles on útiles para la comparación entre etiquetas 0 y 1.
    """
    res = list()
    
    # Genero la iteración por filas, incluyendo la última
    for col in range(cota_inferior_vertical,
                     cota_superior_vertical):
        lista_actual = list()
        for pos in range(cota_inferior_horizontal,
                         cota_superior_horizontal):
            lista_actual.append('pixel' + str(28*col + pos+1))
        res.append(lista_actual)        
    
    return res
    
    
if __name__ == '__main__':
    # Seccion para posibles usos o pruebas
    pass
