'''
Materia     : Laboratorio de datos - FCEyN - UBA
Autores     : Augusto Gantes, Martin Belmes y Matias D'Andrea
Detalle     : 
    Este archivo contiene funciones vinculadas al desarrollo de los modelos
    utilizados para analizar los datos.    
Creacion    : 6/10/2023
Modificacion: 6/11/2023
'''

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def iteracion_posiciones(df: pd.DataFrame, posiciones: list) -> dict:
    '''
    Esta función se encarga de realizar el proceso de separación de datos,
    entrenamiento y evaluación de cada modelo generado según los tres 
    píxeles seleccionados de una lista de listas.
    La función retorna un dicc de modelos entrenados con los mejores 
    parametros según los píxeles seleccionados junto a los píxeles y 
    el score de validación.
    '''
    
    # Iteramos por lista de posiciones, c\cjto contiene 3 labels
    res = dict()
    cuenta: int = 0  # contador para etiquetar modelos
    for cjto in posiciones:
        
        # Separamos features de target
        X = df.drop(columns='label')  
        X = X[cjto]  # seleccionamos solo 3 píxeles
        Y = df[['label']]
        
        # Separamos la data en dev y val
        X_dev , X_val , Y_dev , Y_val = train_test_split(
            X,Y,test_size=0.1,random_state=0,stratify=Y)
        del X, Y  # Borramos la data que ya no usamos
        
        # Creamos el clasificador junto a un dicc de hiperparametros
        hiper_parametros = {
            'n_neighbors': [i for i in range(3,20)]
            }    
        knn_classifier = KNeighborsClassifier()
        
        # Buscamos los mejores parametros con un gridsearch
        clf = GridSearchCV(knn_classifier, hiper_parametros)
        clf.fit(X_dev, Y_dev)
        
        # Vemos el resultado de generar una estimación entorno a los datos 
        # de validación con 5 pliegues estratificados
        score_val = cross_val_score(clf, X_val, Y_val, cv=5).mean()
        
        # Agregamos los datos que nos parecen importantes al resultado
        res[f'modelo {cuenta}'] = [cjto, clf, score_val]
        cuenta += 1
        
    return res
        
        
if __name__ == '__main__':
    # Seccion para posibles usos o pruebas
    pass
    