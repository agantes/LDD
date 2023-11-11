'''
Materia     : Laboratorio de datos - FCEyN - UBA
Autores     : Augusto Gantes, Martin Belmes y Matias D'Andrea
Detalle     : 
    Este archivo contiene funciones vinculadas al desarrollo de los modelos
    utilizados para analizar los datos.    
Creacion    : 6/10/2023
Modificacion: 11/11/2023
'''

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def iteracion_posiciones(X: pd.DataFrame, Y: pd.DataFrame, 
                         posiciones: list) -> list:
    '''
    Esta función se encarga de realizar el proceso de separación de datos,
    entrenamiento y evaluación de cada modelo generado según los tres 
    píxeles seleccionados de una lista de listas.
    La función retorna una lista de modelos entrenados con los mejores 
    parametros según los píxeles seleccionados junto a los píxeles y 
    el score de validación.
    '''
    
    # Iteramos por lista de posiciones, c\cjto contiene 3 labels
    res = list()
    for cjto in posiciones:
        # Seleccionamos solo 3 píxeles
        X_cjto = X[cjto]
        
        # Creamos el clasificador junto a un dicc de hiperparametros
        hiper_parametros = {
            'n_neighbors': [i for i in range(3,20)]
            }    
        knn_classifier = KNeighborsClassifier()
        
        # Buscamos los mejores parametros con un gridsearch
        clf = GridSearchCV(knn_classifier, hiper_parametros)
        clf.fit(X_cjto, Y)
        
        # Agregamos los datos que nos parecen importantes al resultado
        res.append([cjto, clf])
        
    return res
        
        
if __name__ == '__main__':
    # Seccion para posibles usos o pruebas
    pass
    