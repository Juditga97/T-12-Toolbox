# Team Challenge — Toolbox de Machine Learning

## 📌 Descripción

Este repositorio contiene el desarrollo de una **toolbox de funciones en Python** para apoyar el análisis exploratorio de datos y la selección de variables en problemas de Machine Learning, así como su aplicación sobre el dataset Iris.

El objetivo es crear funciones reutilizables que faciliten el preprocesamiento de los datos antes del entrenamiento de modelos.

---

## 📁 Estructura del repositorio

- `toolbox_ML.py`  
  Archivo con las funciones desarrolladas para el análisis de variables y selección de features.

- `Team_Challenge_ToolBox.ipynb`  
  Notebook con el enunciado del Team Challenge proporcionado por el bootcamp.

- `main.ipynb`  
  Notebook principal donde se aplican las funciones de la toolbox al dataset Iris.

- `data`  
  Carpeta donde se encuentra el dataset utilizado para las pruebas (clasificación de flores Iris).

- `funciones_individuales`
  Carpeta donde se encuentran todas las funciones del equipo en sus respectivos notebooks.

---

## 🧰 Funcionalidades de la toolbox

La toolbox incluye funciones para:

- Analizar tipos de variables y cardinalidad
- Tipificar variables (numéricas y categóricas)
- Seleccionar variables relevantes para regresión
- Analizar relación entre variables categóricas y la variable objetivo
- Visualizar relaciones entre variables y target

Estas funciones permiten automatizar parte del análisis exploratorio previo al modelado.

---

### ▶️ Cómo usar el proyecto

1. Cargar el dataset `iris.csv`.
2. Abrir el archivo `main`.
3. Importar las librerías necesarias y el módulo `toolbox_ML.py`.
4. Ejecutar los notebooks para ver ejemplos de uso de cada función.

### Ejemplo de importación y uso

```python
import toolbox_ML as tml

tml.tipifica_variables(df)
tml.describe_df(df)
```

## 👥 Autores

- Judit García  
- Javier López  
- Sara Ruiz  
- Jennifer Sotelo
- Patricia García  

