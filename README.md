# Clasificación de enfermedad cardíaca — Rúbrica 3

**Estudiante:** Sebastian Navarro Rojas  
**Asignatura:** Minería de Datos · Universidad de la Costa  
**Fuente:** `patient_dataset.csv`, suministrado por el docente a través de Moodle.

## Objetivo
Desarrollar y evaluar, con fines académicos, un proceso de clasificación binaria de enfermedad cardíaca a partir de 15 variables. El proyecto incluye análisis exploratorio, preparación de datos, comparación de algoritmos, evaluación del modelo y una aplicación Streamlit que muestra la clase predicha y una probabilidad estimada.

## Archivos
- `app.py`: aplicación web desarrollada con Streamlit.
- `patient_dataset.csv`: conjunto de datos de 6.000 registros.
- `pipeline_heart_disease.joblib`: pipeline serializado con preprocesamiento y Árbol de Decisión.
- `rubrica3_auditoria_final.ipynb`: notebook disponible actualmente en el repositorio, con entrenamiento, evaluación y auditoría.
- `requirements.txt`: dependencias para ejecutar la aplicación, incluida scikit-learn 1.6.1.

## Ejecutar localmente
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Resultados del modelo
El modelo seleccionado fue un **Árbol de Decisión**, optimizado por F1-score mediante validación cruzada. En los 1.200 registros de prueba obtuvo:

| Métrica | Resultado |
|---|---:|
| Accuracy | 48,67 % |
| Precision | 49,20 % |
| Recall | 93,14 % |
| F1-score | 64,39 % |
| Balanced accuracy | 48,81 % |
| ROC-AUC | 47,83 % |

**Matriz de confusión** (filas: valor real; columnas: predicción):

```text
[[ 27, 575],
 [ 41, 557]]
```

El modelo produjo 575 falsos positivos y 41 falsos negativos. La auditoría mediante permutación de etiquetas con Regresión Logística obtuvo un valor p aproximado de 0,4194. Estos resultados no respaldan una capacidad predictiva suficiente para uso clínico.

## Pruebas funcionales documentadas
1. **Entrada válida:** el formulario generó clase 1 con probabilidad estimada de 50,2 %.
2. **Entrada inválida:** al intentar ingresar edad -5 se observó una indicación visual de valor fuera del rango. Esta evidencia no comprueba por sí sola que el envío del formulario estuviera bloqueado.
3. **Consistencia con Google Colab:** para el registro 3292, tanto el notebook como Streamlit predijeron clase 1; la aplicación mostró 56,6 %.
4. **Valor límite:** con edad 18 y los demás valores predeterminados, la aplicación mostró clase 1 y 50,2 %.
5. **Acceso público:** en una ventana de incógnito se ejecutó una predicción con resultado de clase 1 y 50,2 %.

Estas pruebas comprueban aspectos técnicos del funcionamiento, pero no validan la precisión clínica del modelo.

## Aplicación publicada
**Streamlit:** https://prediccion-enfermedad-cardiaca-i7dgxqmx6cvyhgjnwgxdtj.streamlit.app/

**Repositorio:** https://github.com/sebasnr13/prediccion-enfermedad-cardiaca

La aplicación contiene secciones de exploración de datos, predicción y evaluación de limitaciones.

## Consideraciones sobre los datos y el uso responsable
El archivo CSV fue suministrado por el docente. **Está publicado en este repositorio público; se debe confirmar que su distribución pública está autorizada.** Si no lo está, corresponde retirarlo del acceso público y acordar con el docente una forma adecuada de compartir o evaluar el proyecto.

**Aviso:** este es un prototipo académico sin validez clínica. No utilizar sus predicciones para diagnóstico, tratamiento ni decisiones médicas. Evitar ingresar datos personales identificables.

## Uso de inteligencia artificial
Se utilizó ChatGPT como apoyo para organizar la documentación, explicar procedimientos y resultados, y complementar la discusión del proyecto. Las respuestas de IA requieren contrastarse con el notebook y las pruebas realizadas; no sustituyen la evaluación ni la responsabilidad académica del estudiante.
