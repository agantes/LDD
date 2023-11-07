'''
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
Modificacion: 3/11/2023
'''

# %% Importacion de librerias
import pandas as pd
import funciones_analisis as fa  # funciones de analisis
import funciones_modelos as fm  # funciones de modelos
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Carga de Datos
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
    fa.plot_maps(mat_std_etiqueta,
                 f'Desvío estándar por Píxel etiqueta {etiqueta}',
                 mat_mean_etiqueta,
                 f'Promedio por Píxel etiqueta {etiqueta}')

# Se puede apreciar que según distintos tipos de prenda cambia bastante la
# forma de los gráficos, hasta se puede distinguir si de que clase son según
# la forma de los datos, lo que indica que hay cierto grado de acierto en lo
# que estamos analizando. Utilizar estas imagenes para comparación intraclase.

# Genero un gráfico que permita comparar todas las clases en paralelo
# con sus imagenes promedio.
fa.plot_promedios_clases(df)

# Eliminamos variables de uso temporal de esta sección
del etiquetas, etiqueta, df_etiqueta
del df_etiqueta_mean_pixel, df_etiqueta_std_pixel
del mat_mean_etiqueta, mat_std_etiqueta

# %%% Analisis auxiliar sobre etiquetas 0 y 1

# Sabemos que todas las clases estan balanceadas, todas poseen 6000 imagenes
# Analizamos que píxeles son útiles según std para diferenciar las clases
# Seleccionamos la info correspondiente a cada etiqueta
etiqueta0 = df[df['label'] == 0].drop('label', axis=1)
etiqueta1 = df[df['label'] == 1].drop('label', axis=1)

# Pasamos la información al std de cada píxel
std_etiqueta0 = fa.std_pixeles(etiqueta0, label=False)
std_etiqueta1 = fa.std_pixeles(etiqueta1, label=False)

# Generamos los np.arrays de cada etiqueta
mat_std_etiqueta0 = fa.map_metrica(std_etiqueta0, 'std')
mat_std_etiqueta1 = fa.map_metrica(std_etiqueta1, 'std')

# Analizamos por separado la std
fa.plot_map(mat_std_etiqueta0, 'Desvío Estándar etiqueta 0')
fa.plot_map(mat_std_etiqueta1, 'Desvío Estándar etiqueta 1')

# Parece buena idea elegir los pixeles que se encuentran en el medio de los
# pantalones que parecen poseer entre 20 y 40 de std
# Son los pixeles que mas varian para una remera
# Vemos como localizar estos pixeles

# Separo la parte de interes a prueba y error
cotas = (14, 28, 13, 16)  # poiciones según min, max y ver, hor
rdi0 = mat_std_etiqueta0[cotas[0]:cotas[1], cotas[2]:cotas[3]] 
rdi1 = mat_std_etiqueta1[cotas[0]:cotas[1], cotas[2]:cotas[3]] 
fa.plot_map(rdi0, 'Área de Interes\netiqueta 0')
fa.plot_map(rdi1, 'Área de Interes\netiqueta 1')

# Observamos que en la clase 0 la zona de interes es sumamente uniforme 
# mientras que en la clase 1 hay una sección de baja std, hacemos foco en esa
# zona
# Recuperamos las posiciones
lista_posiciones = fa.recuperar_posciciones(*cotas)

# Eliminamos variables de analisis, posiblemente reciclemos lista_posiciones
del etiqueta0, etiqueta1, std_etiqueta0, std_etiqueta1
del mat_std_etiqueta0, mat_std_etiqueta1
del cotas, rdi0, rdi1, lista_posiciones  

# %% KNN sobre clases 0 y 1

# Separo el dataframe, me quedo solo con pantalones y remeras (labels 0 y 1)
df_knn = df[(df['label'] == 0) | (df['label'] == 1)]

# Basandonos en el analisis auxiliar de la etiqueta 0 y 1, entrenamos al
# clasificador con diferentes sets de 3 pixeles según las regiones de interes
# encontrada
cotas = (14, 28, 13, 16) 
lista_posiciones = fa.recuperar_posciciones(*cotas)
modelos_knn = fm.iteracion_posiciones(df_knn, lista_posiciones)

# El random_state es 0 para el train_test_split
# Buscamos aquel que tenga el mayor cross_val_score
max_puntaje: float = 0 
for modelo in modelos_knn:
    if modelo[2] > max_puntaje:
        max_posiciones = modelo[0]
        max_modelo = modelo[1]
        max_puntaje = modelo[2]
print('Mejor tripla:', max_posiciones)
print('Mayor puntaje:', max_puntaje)
print('Parametros:', max_modelo.best_params_)

# Según la última ejecución, el mejor parametro es un k de 8 con el uso de 
# píxeles intermedios de la sección de interes seleccionada (630, 631, 632)
# El puntaje es de 0.97, aproximadamente

# %%% Descarte Analisis KNN

# Borramos los datos que no son relevantes para la siguiente sección
del df_knn, cotas, lista_posiciones, modelos_knn
del max_modelo, max_puntaje, max_posiciones, modelo

# %% Clasificación multiclase

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
