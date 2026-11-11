# Clasificación de enfermedad cardíaca — Rúbrica 3

**Estudiante:** Sebastian Navarro Rojas  
**Asignatura:** Minería de Datos · Universidad de la Costa  
**Fuente:** `patient_dataset.csv`, entregado por el docente a través de Moodle.

## Objetivo
Demostrar un proceso académico de clasificación binaria, desde el análisis exploratorio hasta una aplicación Streamlit que recibe 15 variables y muestra la clase y probabilidad calculadas por un pipeline de scikit-learn.

## Archivos
- `app.py`: aplicación Streamlit.
- `patient_dataset.csv`: conjunto de datos (6.000 registros).
- `pipeline_heart_disease.joblib`: pipeline serializado (preprocesamiento y Árbol de Decisión).
- `rubrica3_auditoria_final.ipynb`: entrenamiento, evaluación y auditoría.
- `requirements.txt`: dependencias, incluyendo scikit-learn 1.6.1.

## Ejecutar localmente
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Resultados y límites
Accuracy 48,67 %; precision 49,20 %; recall 93,14 %; F1 64,39 %; ROC-AUC 47,83 %. La matriz de confusión del conjunto de prueba fue `[[27,575],[41,557]]`. La auditoría con etiquetas permutadas obtuvo p ≈ 0,4194. Los resultados son próximos al azar y no tienen validez clínica.

**Aviso:** prototipo académico. No usar para diagnóstico ni decisiones médicas. Los valores ingresados no se guardan en una base de datos; evita introducir información personal identificable.

## Pruebas recomendadas
1. Caso válido: completar formulario y verificar clase y probabilidad.
2. Caso inválido: intentar ingresar un número fuera del rango permitido; el formulario debe impedirlo.
3. Caso conocido: tomar una fila de `X_test` del notebook, ingresar sus valores y comprobar que la clase y probabilidad coinciden con la predicción del mismo pipeline. Documentar captura y resultados.

## Publicación
Crear repositorio GitHub, cargar estos archivos y desplegar en Streamlit Community Cloud seleccionando `app.py` como archivo principal. **Antes de publicar el CSV**, confirmar que el docente permite compartirlo públicamente. Si no, solicitar autorización o usar un repositorio/despliegue apropiado sin publicar datos restringidos.

## Uso de IA
Se utilizó ChatGPT como apoyo para estructurar el notebook, revisar el preprocesamiento, proponer auditoría de señal predictiva y generar la aplicación Streamlit y documentación. Los resultados se revisaron frente a las salidas del notebook; la IA no reemplaza la validación del estudiante ni garantiza el desempeño del modelo.
