"""
Materia     : Laboratorio de datos - FCEyN - UBA
Autores     : Augusto Gantes, Martin Belmes y Matias D'Andrea
Detalle     : 
    Este archivo contiene solamente funciones relacionadas con la exploración de 
    los datos.    
Creacion    : 27/10/2023
Modificacion: 27/10/2023
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
    fig, ax = plt.figure(figsize=(4,4))
    plt.imshow(mat, cmap='magma', origin='upper')
    plt.title(title)
    plt.colorbar(shrink=0.8)
    plt.show()
    plt.close()

def plot_maps(mat1: np.array, title1:str,
              mat2: np.array, title2:str) -> None:
    '''
    Genera un plot en paralelo de dos np.arrays basandose en las otras
    funciones.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im1 = axes[0].imshow(mat1, cmap='inferno', origin='upper')
    axes[0].set_title(title1)
    plt.colorbar(im1, shrink=.7, ax=axes[0])
    im2 = axes[1].imshow(mat2, cmap='inferno', origin='upper')
    axes[1].set_title(title2)
    plt.colorbar(im2, shrink=.7, ax=axes[1])
    plt.show()
    plt.close()

# def plot_promedios_clases
    
if __name__ == '__main__':
    # Seccio=ón para posibles usos o pruebas
    pass
