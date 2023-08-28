import os
import csv

def leer_parque(nombre_archivo: str, parque: str) -> list[dict]:
    """
    Gnera una lista de diccionarios con el nombre de un archivo y un parque
    que este contenido en el archivo. El param nombre_archivo debe contener
    el directorio en caso de ser ejecutado desde un proyecto de spyder cuya
    solución albergue inmediatamente el archivo. Se asigna automáticamente 
    los nombres con la primera fila.
    """ 
    res: list = []
    with open(nombre_archivo, encoding="utf8", newline="") as cvsfile:
        freader = csv.DictReader(cvsfile, delimiter=",")
        for row in freader:
            if row["espacio_ve"] == parque:
                res.append(row)
    return res

def especies(lista_arboles: list) -> set:
    """
    Con una lista de diccionarios como la provista por leer_parque
    genera un set con las especies de los arboles contenidos en la lista.
    """
    res: set = set()
    for arbol in lista_arboles:
        res.add(arbol["nombre_com"])
    return res

def contar_ejemplares(lista_arboles: list) -> dict:
    """
    A partir de una lista de arboles generada por leer_parque cuenta
    la cantidad de ejemplares de cada especie, según nombre_com, y
    devuelve un diccionario con el nombre de la especie y el recuento.
    """
    res: dict = {}
    for arbol in lista_arboles:
        nom: str = arbol["nombre_com"]
        res[nom] = res.get(nom, 0)  # asignar el nombre al dict
        res[nom] += 1  # sumar la aparición
    return res

if __name__ == "__main__":
    # conseguimos lista de rows del parque GENERAL PAZ
    directorio: str = "Proyecto_Arboles"
    archivo: str = "arbolado-en-espacios-verdes.csv"
    nombre_archivo: str = os.path.join(directorio, archivo)
    parque: str = "GENERAL PAZ"
    parque_gral_paz: list[dict] = leer_parque(nombre_archivo, parque)
    
    # vemos las especies del parque
    especies_gral_paz: set = especies(parque_gral_paz)    
    
    # contamos los ejemplares de una especie
    nejemplares_gral_paz: dict = contar_ejemplares(parque_gral_paz)
        