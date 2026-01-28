# Team Challenge — Toolbox de Machine Learning

## 📌 Descripción

Este repositorio contiene el desarrollo de una **toolbox de funciones en Python** para apoyar el análisis exploratorio de datos y la selección de variables en problemas de Machine Learning, así como su aplicación sobre el dataset Iris.

El objetivo es crear funciones reutilizables que faciliten el preprocesamiento de los datos antes del entrenamiento de modelos.

---

## 📁 Estructura del repositorio

- `toolbox_ML.py`  
  Archivo con las funciones desarrolladas para el análisis de variables y selección de features.

- `Team_Challenge_ToolBox.ipynb`  
  Notebook donde se prueban y se explican las funciones utilizando el dataset Iris.

- `main.ipynb`  
  Notebook principal de trabajo con el dataset y aplicación de la toolbox.

- `iris.csv`  
  Dataset utilizado para las pruebas (clasificación de flores Iris).

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

## ▶️ Cómo usar el proyecto

1. Cargar el dataset `iris.csv`
2. Importar las funciones desde `toolbox_ML.py`
3. Ejecutar los notebooks para ver ejemplos de uso de cada función

Ejemplo de importación:

```python
from toolbox_ML import tipifica_variables, describe_df
```
---

## 👥 Autores

- Judit García  
- Javier López  
- Sara Ruiz  
- Jennifer Sotelo
- Patricia García  

