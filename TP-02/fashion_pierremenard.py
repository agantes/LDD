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
Modificacion: 25/10/2023
"""

#%% Importacion de librerias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

#%% Carga de Datos
df = pd.read_csv('fashion-mnist.csv', encoding='utf-8')

#%% Funciones 

#%% Analisis de los Datos

# Para empezar analizamos la cantidad de datos y descriptores básicos
df.head()
df.describe()  # el maximo y cuartiles varian bastante entre posiciones
df.info()

#%%% Varianza por Pixel

# Segun lo que vemos de lo variable que es la data segun posicion de pixel
# nos preguntamos si es que hay posiciones que siempre sean de una misma 
# intensidad en la escala de grises, i.e., puede que no haya varianza
posiciones = list(set(df.columns) - {'label'})
varianza_por_posicion = {'posicion': [], 'varianza': []}
for posicion in posiciones:
    numero_posicion_str = re.findall(r'\d+', posicion)  # esto devuelve lista
    numero_posicion = int(numero_posicion_str[0])
    varianza_por_posicion['posicion'].append(numero_posicion)
    varianza_por_posicion['varianza'].append(df[posicion].std())
varianza_por_posicion = pd.DataFrame(varianza_por_posicion)

# Vemos como se distribuyen las disperciones
fig, ax = plt.subplots(figsize=(10,6))
sns.kdeplot(data=varianza_por_posicion, x='posicion', y='varianza',
            color='black', alpha=.5)
sns.scatterplot(data=varianza_por_posicion, x='posicion', y='varianza')
ax.set_xlabel('Pixel', fontsize=18)
ax.set_ylabel('Varianza', fontsize=18)
plt.title('Varianza según Pixel', fontsize=20)
plt.grid()
plt.show()
plt.close()

# Se observan muchas variables con poca variacion, incluso sin haber separado 
# por categoria. Tambien podemos ver que hay un cumulo de valores en la parte
# inferior izquierda que contiene valores con varianza por debajo de 20.

# Descarto variables que se usaron para graficar y computar
del fig, ax
del varianza_por_posicion, posiciones
del posicion, numero_posicion_str, numero_posicion

#%%% Separacion por etiqueta
# Vemos el recuento de clasificaciones 
df['label'].value_counts()  # es uniforme, 6000 entradas para las 10 etiquetas


