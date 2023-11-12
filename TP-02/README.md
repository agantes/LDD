# Funcionamiento entrega

La entrega tiene 3 archivos funcionales junto con una carpeta de proyecto de spyder. Esta el archivo con el nombre del grupo y dos archivos auxiliares que contienen funciones. El código del archivo principal esta separado en bloques con la intención de ser ejecutados en orden sucesivo. 

El env utilizado deberá contener las bibliotecas especificadas en la sección de bibliotecas y se deberá de usar el env especificado, o cualquier otro que cumpla con los requisitos de manerea identica o similar, en el Spyder IDE.

## Bibliotecas

Las siguientes bibliotecas son necesarias para la ejecución:
* numpy 1.25.2
* pandas 1.5.3
* matplotlib 3.7.1
* scikit-learn 1.2.2
* seaborn 0.12.2
* spyder-kernels 2.5.0

**Aclaración**: el paquete `spyder-kernels` es necesario en dicha versión para la Spyder IDE 5.5.0.

Dejamos un comando para poder generar un `conda env` para correr la entrega.
```
conda create --name pierre_menard_env python=3.11 seaborn=0.12.2 scikit-learn=1.2.2 matplotlib=3.7.1 numpy=1.25.2 pandas=1.5.3
conda activate pierre_menard_env
pip install spyder-kernels==2.5.0
```

Asimismo, también el comando para su desintsalación.
```
conda deactivate pierre_menard_env
conda remove --name pierre_menard_env --all
```
