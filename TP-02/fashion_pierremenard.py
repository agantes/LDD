"""
Materia     : Laboratorio de datos - FCEyN - UBA
Autores     : Augusto Gantes, Martin Belmes y Matias D'Andrea
Detalle     : 
    Este archivo contiene el desarrollo de todas las actividades dispuestas 
    en el TP-02 de manera tal que se ejecute el código por secciones. Se 
    comentaran los hallazgos a lo largo del codigo. Decidir que hacer con 
    funciones auxiliares, si es3 que usamos.
    
    Importante: se asume que el proyecto de Spyder esta situado en la carpeta
    en la que se encuentran todos los archivos. NO entregar dataset.
    
Creacion    : 25/10/2023
Modificacion: 26/10/2023
"""

# %% Importacion de librerias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import funciones_analisis as fa  # funciones de analisis

# %% Carga de Datos
df = pd.read_csv('fashion-mnist.csv', encoding='utf-8')

# %% Analisis de los Datos

# Para empezar analizamos la cantidad de datos y descriptores básicos
df.head()
df.describe()  # el maximo y cuartiles varian bastante entre posiciones
df.info()

# %%% Metricas por Pixel (General)

# Segun lo que vemos de lo variable que es la data segun posicion de pixel
# nos preguntamos si es que hay posiciones que siempre sean de una misma
# intensidad en la escala de grises, i.e., puede que no haya std.
# Modifico el df para analizar std p/pixel y el promedio de los pixeles
std_por_posicion = fa.std_pixeles(df)
mean_por_posicion = fa.mean_pixeles(df)

# Genero las matrices de desviación estándar y promedio
mat_std = fa.map_metrica(std_por_posicion, 'std')
mat_mean = fa.map_metrica(mean_por_posicion, 'mean')

# Gráfico
fa.plot_maps(mat_std, 'Desviación Estándar por Píxel',
             mat_mean, 'Promedio por Píxel')

# Se observa que ciertos bordes tienen poca desviación estándar en relación al
# resto de los píxeles. Podemos elegir una cota para eliminar ciertos datos
# según su desvío estándar en las secciones dedicadas a inferencia.

# Descarto variables que se usaron para graficar y computar
del std_por_posicion, mean_por_posicion, mat_mean, mat_std

# %%% Separacion por etiqueta
# Vemos el recuento de clasificaciones
df['label'].value_counts()  # es uniforme, 6000 entradas para las 10 etiquetas

# Vemos si realmente el dataset contiene informacion diferente para cada label
etiquetas = df['label'].unique()
etiquetas.sort()

# Vemos que sucede con cada prenda
for etiqueta in etiquetas:
    # Filtramos por etiqueta
    df_etiqueta = df[df['label'] == etiqueta]
    df_etiqueta = df_etiqueta.drop('label', axis=1)

    # Calculamos desviación y media junto a sus matrices
    df_etiqueta_std_pixel = fa.std_pixeles(df_etiqueta, label=False)
    df_etiqueta_mean_pixel = fa.mean_pixeles(df_etiqueta, label=False)
    mat_std_etiqueta = fa.map_metrica(df_etiqueta_std_pixel, 'std')
    mat_mean_etiqueta = fa.map_metrica(df_etiqueta_mean_pixel, 'mean')
    
    # Graficamos
    fa.plot_maps(mat_std_etiqueta, 'Desvío estándar por Píxel',
                 mat_mean_etiqueta, 'Promedio por Píxel')
    
# Se puede apreciar que según distintos tipos de prenda cambia bastante la
# forma de los gráficos, hasta se puede distinguir si de que clase son según
# la forma de los datos, lo que indica que hay cierto grado de acierto en lo 
# que estamos analizando. Utilizar estas imagenes para comparación intraclase.
