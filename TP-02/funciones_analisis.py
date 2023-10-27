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

if __name__ == '__main__':
    # Seccio=ón para posibles usos o pruebas
    pass
