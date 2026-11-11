from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Clasificación de enfermedad cardíaca | Proyecto académico', page_icon='🫀', layout='wide')
st.title('🫀 Clasificación de enfermedad cardíaca')
st.caption('Proyecto académico de Minería de Datos · Sebastian Navarro Rojas · Universidad de la Costa')
st.warning('⚠️ Prototipo académico sin validez clínica. No debe utilizarse para diagnóstico, tratamiento ni decisiones médicas.')
st.write('**Objetivo:** demostrar un flujo de minería de datos que permite explorar el conjunto de datos y ejecutar un modelo de clasificación binaria. **Fuente:** patient_dataset.csv suministrado por el docente a través de Moodle.')

@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / 'patient_dataset.csv')

@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / 'pipeline_heart_disease.joblib')

try:
    df = cargar_datos()
    modelo = cargar_modelo()
except Exception as exc:
    st.error(f'No se pudieron cargar los archivos del proyecto: {exc}')
    st.info('Comprueba que patient_dataset.csv y pipeline_heart_disease.joblib estén junto a app.py y que scikit-learn sea 1.6.1.')
    st.stop()

pestanas = st.tabs(['📊 Exploración de datos', '🔎 Predicción', '📈 Evaluación y limitaciones'])

with pestanas[0]:
    st.subheader('Análisis exploratorio')
    a, b, c = st.columns(3)
    a.metric('Registros', f'{len(df):,}')
    b.metric('Variables predictoras', len(df.columns)-1)
    c.metric('Valores ausentes', int(df.isna().sum().sum()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df['heart_disease'].value_counts().sort_index().plot.bar(ax=axes[0], color=['#60a5fa','#f97316'])
    axes[0].set_title('Distribución de la variable objetivo')
    axes[0].set_xlabel('heart_disease (0 / 1)')
    axes[0].set_ylabel('Pacientes')
    df.boxplot(column='age', by='heart_disease', ax=axes[1], grid=False)
    axes[1].set_title('Edad por clase')
    axes[1].set_xlabel('heart_disease')
    axes[1].set_ylabel('Edad')
    df.boxplot(column='cholesterol', by='heart_disease', ax=axes[2], grid=False)
    axes[2].set_title('Colesterol por clase')
    axes[2].set_xlabel('heart_disease')
    axes[2].set_ylabel('Colesterol')
    fig.suptitle('')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Gráfica 1: clases casi equilibradas. Gráficas 2 y 3: permiten comparar distribuciones por clase, sin demostrar por sí solas relaciones causales.')
    with st.expander('Diccionario de columnas y valores ausentes'):
        st.dataframe(pd.DataFrame({'Columna':df.columns,'Tipo':df.dtypes.astype(str).values,'Ausentes':df.isna().sum().values}), hide_index=True)

with pestanas[1]:
    st.subheader('Formulario de paciente')
    st.write('Introduce valores de demostración. El formulario usa los mismos nombres de variables que el pipeline entrenado. Los campos opcionales vacíos se imputan mediante el pipeline.')
    features = df.drop(columns=['heart_disease'])
    cat_cols = ['gender','chest_pain_type','exercise_angina','hypertension','residence_type','smoking_status']
    numeric_cols = [c for c in features if c not in cat_cols]
    registro = {}
    with st.form('formulario'):
        left, right = st.columns(2)
        for i, col in enumerate(features.columns):
            target_col = left if i % 2 == 0 else right
            with target_col:
                s = features[col]
                if col in cat_cols:
                    opciones = sorted(s.dropna().unique().tolist(), key=lambda x: str(x))
                    if s.isna().any():
                        opciones = ['Sin dato'] + opciones
                    eleccion = st.selectbox(col, opciones, format_func=lambda x: str(x), key=col)
                    registro[col] = np.nan if eleccion == 'Sin dato' else eleccion
                else:
                    default = float(s.median())
                    minimo = float(s.min())
                    maximo = float(s.max())
                    # Rango exploratorio, no rango clínico validado
                    registro[col] = st.number_input(col, min_value=minimo, max_value=maximo, value=default, step=1.0 if pd.api.types.is_integer_dtype(s) else 0.1, key=col)
        ejecutar = st.form_submit_button('Calcular predicción', type='primary')
    if ejecutar:
        entrada = pd.DataFrame([registro], columns=features.columns)
        try:
            pred = int(modelo.predict(entrada)[0])
            st.subheader('Resultado del modelo')
            st.metric('Clase predicha', f'{pred} — '+('Con enfermedad (según modelo)' if pred == 1 else 'Sin enfermedad (según modelo)'))
            if hasattr(modelo, 'predict_proba'):
                probs = modelo.predict_proba(entrada)[0]
                clases = list(modelo.classes_)
                if 1 in clases:
                    prob = float(probs[clases.index(1)])
                    st.write(f'**Probabilidad estimada de clase 1:** {prob:.1%}')
                    st.progress(prob)
                    st.caption('Probabilidad producida por el modelo; no es una probabilidad clínica validada ni una estimación fiable del riesgo individual.')
            st.warning('La exactitud del modelo en prueba fue inferior al 50 %. Esta salida solo demuestra el funcionamiento del pipeline.')
        except Exception as exc:
            st.error(f'Error al realizar la predicción: {exc}')

with pestanas[2]:
    st.subheader('Resultados del conjunto de prueba')
    st.caption('Valores registrados en el notebook de la tercera rúbrica (no calculados nuevamente por la aplicación).')
    valores = {'Accuracy':0.4867,'Precision':0.4920,'Recall':0.9314,'F1-score':0.6439,'Balanced accuracy':0.4881,'ROC-AUC':0.4783}
    cols = st.columns(3)
    for i, (k,v) in enumerate(valores.items()):
        cols[i%3].metric(k, f'{v:.2%}')
    st.write('**Modelo seleccionado:** Árbol de Decisión (optimización por F1 en validación cruzada).')
    st.write('**Matriz de confusión** — filas: clase real, columnas: clase predicha.')
    cm = np.array([[27,575],[41,557]])
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap='Blues')
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=15)
    ax.set_xticks([0,1],['0','1']);ax.set_yticks([0,1],['0','1'])
    ax.set_xlabel('Predicción');ax.set_ylabel('Valor real')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('''**Interpretación:** el modelo detectó 557 positivos verdaderos, pero generó 575 falsos positivos y 41 falsos negativos. La exactitud fue inferior a la referencia Dummy (~50,19 %). El ROC-AUC fue cercano a 0,5, lo que no demuestra discriminación útil. En la auditoría, la prueba de permutación obtuvo un valor p aproximado de 0,4194; no se halló evidencia suficiente de señal predictiva superior al azar.\n\n**Limitaciones:** valores ausentes, registros inverosímiles (por ejemplo, presión arterial igual a cero), etiquetas y procedencia pendientes de validación con el docente, y desempeño insuficiente para cualquier uso clínico. El modelo no sustituye a un profesional de salud.''')
    st.info('Pruebas para la entrega: (1) paciente con datos válidos, (2) entrada inválida rechazada por los controles, (3) registro conocido del conjunto de prueba y comparación con el notebook.')
