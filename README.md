# Sistema de Predicción de Deserción Académica

## Descripción

Este proyecto desarrolla un sistema de predicción del estado académico de estudiantes universitarios mediante técnicas de aprendizaje estadístico. El modelo fue implementado en Python utilizando Scikit-Learn, entrenado en Google Colab y desplegado como una aplicación web mediante Streamlit Community Cloud.

## Objetivo

Predecir el estado académico de un estudiante universitario, clasificándolo en una de las siguientes categorías:

- Dropout (Deserción)
- Enrolled (Matriculado)
- Graduate (Graduado)

## Tecnologías utilizadas

- Python
- Google Colab
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Streamlit
- GitHub

## Dataset

Se utilizó el conjunto de datos **Predict Students Dropout and Academic Success**, empleado para el entrenamiento y evaluación del modelo de aprendizaje estadístico.

## Archivos principales

- **app.py** → Aplicación web desarrollada con Streamlit.
- **modelo_desercion.pkl** → Modelo entrenado.
- **scaler.pkl** → Escalador de datos.
- **encoder.pkl** → Codificador de etiquetas.
- **requirements.txt** → Dependencias necesarias para ejecutar el proyecto.

## Despliegue

La aplicación fue publicada mediante **Streamlit Community Cloud**, utilizando GitHub como repositorio del proyecto.

## Autor

Proyecto desarrollado como trabajo final del curso de Aprendizaje Estadístico.
