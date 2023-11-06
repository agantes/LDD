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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import funciones_analisis as fa  # funciones de analisis

# %% Carga de Datos
df = pd.read_csv('fashion-mnist.csv', encoding='utf-8')

# %% Analisis de los Datos

# Para empezar analizamos la cantidad de datos y descriptores básicos
df.head()
df.describe()  # el maximo y cuartiles varian bastante entre posiciones
df.info()

# %%% Desviacion estandar (std) por Pixel

# Segun lo que vemos de lo variable que es la data segun posicion de pixel
# nos preguntamos si es que hay posiciones que siempre sean de una misma
# intensidad en la escala de grises, i.e., puede que no haya std.
std_por_posicion = fa.std_pixeles(df)  # Modifico el df para analizar std p/pixel

# Vemos como se distribuyen las disperciones
fa.plot_std_pixeles(std_por_posicion, 'Desviación Estándar por Píxel')

# Se observan muchas variables con poca variacion, incluso sin haber separado
# por categoria. Tambien podemos ver que hay un cumulo de valores en la parte
# inferior izquierda que contiene valores con std por debajo de 20.

# Genero una matrix de aquellos pixeles que tienen la desviacion estandar por
# debajo de 20
cota = 20
mat_std = fa.map_std(std_por_posicion, cota)
fa.plot_map_std(mat_std, 'Pixeles con Desviación Estándar menor a ' + str(cota))

## Hacemos un plot de las figuras en cjto
fa.plot_dist_mat(std_por_posicion, cota, 
                 'Desviación Estándar por Píxel',
                 'Píxeles con Desviación Estándar \ndebajo de ' + str(cota)
                 )

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
    df_etiqueta_std_pixel = fa.std_pixeles(df_etiqueta)
    fa.plot_std_pixeles(df_etiqueta_std_pixel,
                        'Desviación Estándar por Píxel Etiqueta ' + 
                        str(etiqueta))

    # Vemos los pixeles que "no varian mucho" (en comparacion a otros)
    cota = 20
    mat_etiqueta = fa.map_std(df_etiqueta_std_pixel, cota)
    fa.plot_map_std(mat_etiqueta,
                    'Pixeles con Desviación Estándar baja Etiqueta ' + 
                    str(etiqueta))

# Se puede apreciar que según distintos tipos de prenda cambia bastante la
# forma de los gráficos, hasta se puede distinguir si son calzados o no según
# la forma de los datos, lo que indica que hay cierto grdo de acierto en lo que
# estamos analizando


#%% KNN

# Tenemos desde antes que df = nuestro dataframe
df2 = df
#Separo el dataframe , me quedo solo con pantalones y remeras(labels 0 y 1)
df2 = df2[(df['label']==0)|(df['label']==1)]

#Creo los features y el target
X=df2.drop(columns ='label')
X= X[['pixel406','pixel407','pixel408']]
Y=df2[['label']]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.1,random_state=0,stratify=Y)

#Borramos las variables
del X,Y

#Creamos nuestro KNN clasificador

hyper_parametros = {'n_neighbors':[i for i in range(3,20)]}    

knn_model = KNeighborsClassifier()


clf = GridSearchCV(knn_model, hyper_parametros)#busqueda exhaustiva 
buscar = clf.fit(X_train,Y_train)
buscar.best_params_ #Nos dice que la mejor opcion es k = 12
buscar.best_score_ #Da un 94% de score


#%% Clasificación multiclase


x1=df.drop(columns ='label')
y1=df[['label']]


x1_train , x1_test , y1_train , y1_test = train_test_split(x1,y1,test_size=0.1,random_state=0,stratify=y1)

tree = DecisionTreeClassifier(random_state=0) #creo un arbol de tipo gini con altura 5

hyper_params = {'criterion' : ["gini", "entropy"],
                'max_depth' : [4,5,6,7,8,9,10,11,12,13,14] }

clf = GridSearchCV(tree, hyper_params)#busqueda exhaustiva 

buscar = clf.fit(x1_train,y1_train)
buscar.best_params_
buscar.best_score_

cross_val_score(clf, x1_train, y1_train, cv=5)
cross_val_score(clf, x1_test, y1_test, cv=5)
