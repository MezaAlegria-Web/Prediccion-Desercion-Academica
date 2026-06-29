import streamlit as st
import pandas as pd
import joblib
import numpy as np

modelo = joblib.load("modelo_desercion.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(
    page_title="Predicción de Deserción Académica",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center;">
        <h1>🎓 Sistema de Predicción de Deserción Académica</h1>
        <p style="font-size:18px;">
        Proyecto de Aprendizaje Estadístico basado en Machine Learning para estimar
        el estado académico de un estudiante universitario.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

si_no = {"No": 0, "Sí": 1}
genero_opciones = {"Femenino": 0, "Masculino": 1}
asistencia_opciones = {"Nocturna": 0, "Diurna": 1}
estado_civil_opciones = {
    "Soltero": 1,
    "Casado": 2,
    "Viudo": 3,
    "Divorciado": 4,
    "Unión de hecho": 5,
    "Separado legalmente": 6
}

st.subheader("📋 Datos principales del estudiante")

col1, col2, col3 = st.columns(3)

with col1:
    marital_status = estado_civil_opciones[
        st.selectbox("Estado civil", list(estado_civil_opciones.keys()))
    ]
    application_mode = st.number_input("Modo de postulación", value=17)
    application_order = st.number_input("Orden de postulación", value=5)
    course = st.number_input("Código de carrera", value=171)
    daytime_evening = asistencia_opciones[
        st.selectbox("Tipo de asistencia", list(asistencia_opciones.keys()))
    ]
    previous_qualification = st.number_input("Calificación previa", value=1)

with col2:
    previous_qualification_grade = st.number_input("Nota de calificación previa", value=122.0)
    admission_grade = st.number_input("Nota de admisión", value=127.3)
    gender = genero_opciones[
        st.selectbox("Género", list(genero_opciones.keys()))
    ]
    age = st.number_input("Edad al matricularse", value=20)
    debtor = si_no[st.selectbox("¿Tiene deuda?", list(si_no.keys()))]
    tuition_fees = si_no[st.selectbox("¿Pagos al día?", list(si_no.keys()))]

with col3:
    scholarship = si_no[st.selectbox("¿Tiene beca?", list(si_no.keys()))]
    displaced = si_no[st.selectbox("¿Estudiante desplazado?", list(si_no.keys()))]
    educational_special_needs = si_no[
        st.selectbox("¿Necesidades educativas especiales?", list(si_no.keys()))
    ]
    international = si_no[st.selectbox("¿Estudiante internacional?", list(si_no.keys()))]
    nationality = st.number_input("Nacionalidad", value=1)

st.markdown("---")

st.subheader("📚 Rendimiento académico")

col4, col5 = st.columns(2)

with col4:
    cu1_credited = st.number_input("Créditos 1er semestre", value=0)
    cu1_enrolled = st.number_input("Unidades inscritas 1er semestre", value=6)
    cu1_evaluations = st.number_input("Evaluaciones 1er semestre", value=6)
    cu1_approved = st.number_input("Unidades aprobadas 1er semestre", value=6)
    cu1_grade = st.number_input("Nota 1er semestre", value=13.0)
    cu1_without = st.number_input("Sin evaluación 1er semestre", value=0)

with col5:
    cu2_credited = st.number_input("Créditos 2do semestre", value=0)
    cu2_enrolled = st.number_input("Unidades inscritas 2do semestre", value=6)
    cu2_evaluations = st.number_input("Evaluaciones 2do semestre", value=6)
    cu2_approved = st.number_input("Unidades aprobadas 2do semestre", value=6)
    cu2_grade = st.number_input("Nota 2do semestre", value=13.0)
    cu2_without = st.number_input("Sin evaluación 2do semestre", value=0)

st.markdown("---")

st.subheader("👨‍👩‍👧 Variables familiares y socioeconómicas")

col6, col7, col8 = st.columns(3)

with col6:
    mother_qualification = st.number_input("Calificación de la madre", value=19)
    father_qualification = st.number_input("Calificación del padre", value=12)

with col7:
    mother_occupation = st.number_input("Ocupación de la madre", value=5)
    father_occupation = st.number_input("Ocupación del padre", value=9)

with col8:
    unemployment = st.number_input("Tasa de desempleo", value=10.8)
    inflation = st.number_input("Tasa de inflación", value=1.4)
    gdp = st.number_input("PIB", value=1.74)

columnas = [
    'Marital Status',
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Previous qualification (grade)',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

valores = [
    marital_status,
    application_mode,
    application_order,
    course,
    daytime_evening,
    previous_qualification,
    previous_qualification_grade,
    nationality,
    mother_qualification,
    father_qualification,
    mother_occupation,
    father_occupation,
    admission_grade,
    displaced,
    educational_special_needs,
    debtor,
    tuition_fees,
    gender,
    scholarship,
    age,
    international,
    cu1_credited,
    cu1_enrolled,
    cu1_evaluations,
    cu1_approved,
    cu1_grade,
    cu1_without,
    cu2_credited,
    cu2_enrolled,
    cu2_evaluations,
    cu2_approved,
    cu2_grade,
    cu2_without,
    unemployment,
    inflation,
    gdp
]

datos = pd.DataFrame([valores], columns=columnas)

st.markdown("---")

if st.button("🔍 Predecir estado académico"):
    datos_scaled = scaler.transform(datos)
    prediccion = modelo.predict(datos_scaled)
    resultado = encoder.inverse_transform(prediccion)[0]

    probabilidades = modelo.predict_proba(datos_scaled)[0]
    probabilidad_maxima = np.max(probabilidades) * 100

    st.subheader("📌 Resultado de la predicción")

    if resultado == "Dropout":
        st.error("🔴 Resultado estimado: Dropout")
        st.write("El estudiante presenta riesgo de deserción académica.")
    elif resultado == "Enrolled":
        st.warning("🟡 Resultado estimado: Enrolled")
        st.write("El estudiante se mantiene inscrito, pero requiere seguimiento académico.")
    else:
        st.success("🟢 Resultado estimado: Graduate")
        st.write("El estudiante presenta alta probabilidad de éxito académico.")

    st.metric("Nivel de confianza del modelo", f"{probabilidad_maxima:.2f}%")
    st.progress(int(probabilidad_maxima))

    st.markdown("### 📊 Probabilidades por clase")

    for clase, prob in zip(encoder.classes_, probabilidades):
        porcentaje = prob * 100
        st.write(f"{clase}: {porcentaje:.2f}%")
        st.progress(int(porcentaje))

    st.markdown("### Datos ingresados")
    st.dataframe(datos)

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:14px;">
        Proyecto Final de Aprendizaje Estadístico<br>
        Sistema de Predicción de Deserción Académica Universitaria<br>
        Desarrollado con Python, Scikit-Learn, GitHub y Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
