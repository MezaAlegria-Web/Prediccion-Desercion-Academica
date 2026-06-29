import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

modelo = joblib.load("modelo_desercion.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(
    page_title="Predicción de Deserción Académica",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<div style="text-align:center;">
    <h1>🎓 Sistema de Predicción de Deserción Académica</h1>
    <p style="font-size:18px;">
    Aplicación web basada en Aprendizaje Estadístico para estimar el estado académico
    de un estudiante universitario.
    </p>
</div>
""", unsafe_allow_html=True)

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

application_mode_opciones = {
    "1ra fase - contingente general": 1,
    "2da fase - contingente general": 17,
    "3ra fase - contingente general": 18,
    "Estudiante internacional": 15,
    "Mayores de 23 años": 39,
    "Transferencia": 42,
    "Cambio de curso": 43,
    "Cambio de institución/curso": 51
}

course_opciones = {
    "Animación y Diseño Multimedia": 171,
    "Ingeniería Informática": 9119,
    "Enfermería": 9500,
    "Turismo": 9254,
    "Gestión": 9147,
    "Servicio Social": 9238,
    "Diseño de Comunicación": 9070,
    "Periodismo y Comunicación": 9773,
    "Educación Básica": 9853,
    "Agronomía": 9003
}

nacionalidad_opciones = {
    "Portuguesa": 1,
    "Alemana": 2,
    "Española": 6,
    "Italiana": 11,
    "Inglesa": 14,
    "Brasileña": 41,
    "Colombiana": 109
}

previous_qualification_opciones = {
    "Educación secundaria": 1,
    "Bachiller universitario": 2,
    "Licenciatura / grado": 3,
    "Maestría": 4,
    "Doctorado": 5,
    "Frecuencia de educación superior": 6,
    "Curso de especialización tecnológica": 39,
    "Técnico superior profesional": 42
}

st.info(
    "Nota: algunas variables familiares como la calificación u ocupación de los padres "
    "se mantienen como códigos numéricos institucionales del dataset."
)

st.subheader("📋 Datos principales del estudiante")

col1, col2, col3 = st.columns(3)

with col1:
    estado_civil_txt = st.selectbox("Estado civil", list(estado_civil_opciones.keys()))
    marital_status = estado_civil_opciones[estado_civil_txt]

    application_mode_txt = st.selectbox("Modo de postulación", list(application_mode_opciones.keys()))
    application_mode = application_mode_opciones[application_mode_txt]

    application_order = st.number_input("Orden de postulación", value=5)

    course_txt = st.selectbox("Carrera", list(course_opciones.keys()))
    course = course_opciones[course_txt]

    asistencia_txt = st.selectbox("Tipo de asistencia", list(asistencia_opciones.keys()))
    daytime_evening = asistencia_opciones[asistencia_txt]

    previous_qualification_txt = st.selectbox(
        "Calificación previa",
        list(previous_qualification_opciones.keys())
    )
    previous_qualification = previous_qualification_opciones[previous_qualification_txt]

with col2:
    previous_qualification_grade = st.number_input("Nota de calificación previa", value=122.0)
    admission_grade = st.number_input("Nota de admisión", value=127.3)

    genero_txt = st.selectbox("Género", list(genero_opciones.keys()))
    gender = genero_opciones[genero_txt]

    age = st.number_input("Edad al matricularse", value=20)

    deuda_txt = st.selectbox("¿Tiene deuda?", list(si_no.keys()))
    debtor = si_no[deuda_txt]

    pagos_txt = st.selectbox("¿Pagos al día?", list(si_no.keys()))
    tuition_fees = si_no[pagos_txt]

with col3:
    beca_txt = st.selectbox("¿Tiene beca?", list(si_no.keys()))
    scholarship = si_no[beca_txt]

    desplazado_txt = st.selectbox("¿Estudiante desplazado?", list(si_no.keys()))
    displaced = si_no[desplazado_txt]

    necesidades_txt = st.selectbox("¿Necesidades educativas especiales?", list(si_no.keys()))
    educational_special_needs = si_no[necesidades_txt]

    internacional_txt = st.selectbox("¿Estudiante internacional?", list(si_no.keys()))
    international = si_no[internacional_txt]

    nationality_txt = st.selectbox("Nacionalidad", list(nacionalidad_opciones.keys()))
    nationality = nacionalidad_opciones[nationality_txt]

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
    'Marital Status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Admission grade',
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
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
    'Unemployment rate', 'Inflation rate', 'GDP'
]

valores = [
    marital_status, application_mode, application_order, course,
    daytime_evening, previous_qualification, previous_qualification_grade,
    nationality, mother_qualification, father_qualification,
    mother_occupation, father_occupation, admission_grade,
    displaced, educational_special_needs, debtor, tuition_fees,
    gender, scholarship, age, international,
    cu1_credited, cu1_enrolled, cu1_evaluations, cu1_approved,
    cu1_grade, cu1_without,
    cu2_credited, cu2_enrolled, cu2_evaluations, cu2_approved,
    cu2_grade, cu2_without,
    unemployment, inflation, gdp
]

datos = pd.DataFrame([valores], columns=columnas)

st.markdown("---")

if st.button("🔍 Predecir estado académico"):
    datos_scaled = scaler.transform(datos)
    prediccion = modelo.predict(datos_scaled)
    resultado = encoder.inverse_transform(prediccion)[0]

    probabilidades = modelo.predict_proba(datos_scaled)[0]
    probabilidad_maxima = np.max(probabilidades) * 100

    fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    st.subheader("📌 Resultado de la predicción")

    if resultado == "Dropout":
        st.error("🔴 Resultado estimado: Dropout")
        st.write(
            "El modelo estima una alta probabilidad de deserción académica. "
            "Se recomienda realizar seguimiento académico, tutoría personalizada "
            "y apoyo institucional oportuno."
        )
    elif resultado == "Enrolled":
        st.warning("🟡 Resultado estimado: Enrolled")
        st.write(
            "El estudiante se mantiene matriculado. Se recomienda continuar con el "
            "monitoreo académico para prevenir posibles riesgos futuros."
        )
    else:
        st.success("🟢 Resultado estimado: Graduate")
        st.write(
            "El estudiante presenta un perfil favorable para culminar satisfactoriamente "
            "sus estudios universitarios."
        )

    st.metric("Nivel de confianza del modelo", f"{probabilidad_maxima:.2f}%")
    st.progress(int(probabilidad_maxima))

    st.markdown("### 📊 Probabilidades por clase")

    for clase, prob in zip(encoder.classes_, probabilidades):
        porcentaje = prob * 100
        st.write(f"{clase}: {porcentaje:.2f}%")
        st.progress(int(porcentaje))

    st.markdown("### 🧾 Resumen del estudiante")

    resumen = pd.DataFrame({
        "Variable": [
            "Estado civil",
            "Modo de postulación",
            "Carrera",
            "Tipo de asistencia",
            "Calificación previa",
            "Nacionalidad",
            "Género",
            "Edad",
            "Tiene deuda",
            "Pagos al día",
            "Tiene beca",
            "Nota de admisión",
            "Nota 1er semestre",
            "Nota 2do semestre",
            "Tasa de desempleo",
            "Tasa de inflación",
            "PIB",
            "Fecha y hora de predicción"
        ],
        "Valor": [
            estado_civil_txt,
            application_mode_txt,
            course_txt,
            asistencia_txt,
            previous_qualification_txt,
            nationality_txt,
            genero_txt,
            age,
            deuda_txt,
            pagos_txt,
            beca_txt,
            admission_grade,
            cu1_grade,
            cu2_grade,
            unemployment,
            inflation,
            gdp,
            fecha_hora
        ]
    })

    st.table(resumen)

st.markdown("---")

st.markdown("""
<div style="text-align:center; font-size:14px;">
    Proyecto Final de Aprendizaje Estadístico<br>
    Sistema de Predicción de Deserción Académica Universitaria<br>
    Desarrollado con Python, Scikit-Learn, GitHub y Streamlit
</div>
""", unsafe_allow_html=True)
