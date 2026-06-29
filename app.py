import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# =========================
# CARGA DEL MODELO
# =========================

modelo = joblib.load("modelo_desercion.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================

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

st.info(
    "Los campos numéricos respetan la codificación original del dataset utilizado "
    "para entrenar el modelo. Esto permite mantener la consistencia de las predicciones."
)

with st.expander("📖 Ver leyenda de códigos del dataset"):

    st.markdown("""
### Estado civil
- 1 = Soltero
- 2 = Casado
- 3 = Viudo
- 4 = Divorciado
- 5 = Unión de hecho
- 6 = Separado legalmente

---

### Tipo de asistencia
- 0 = Nocturna
- 1 = Diurna

---

### Género
- 0 = Femenino
- 1 = Masculino

---

### Variables binarias

Las siguientes variables utilizan la misma codificación:

- ¿Tiene deuda?
- ¿Pagos al día?
- ¿Tiene beca?
- ¿Estudiante desplazado?
- ¿Necesidades educativas especiales?
- ¿Estudiante internacional?

Codificación:

- 0 = No
- 1 = Sí

---

### Modo de postulación

La variable **Modo de postulación** utiliza códigos oficiales definidos por el dataset original.

Ejemplos:

- 1 = Primera fase
- 15 = Estudiante internacional
- 17 = Segunda fase
- 39 = Mayores de 23 años

Los demás códigos corresponden a modalidades de admisión definidas en el conjunto de datos.

---

### Código de carrera

El valor ingresado corresponde al **código oficial de la carrera universitaria** utilizado en el dataset.

Ejemplos:

- 171 = Animación y Diseño Multimedia
- 9003 = Agronomía
- 9070 = Diseño de Comunicación
- 9085 = Enfermería Veterinaria
- 9119 = Ingeniería Informática
- 9130 = Bioquímica
- 9147 = Gestión
- 9238 = Servicio Social
- 9254 = Turismo
- 9500 = Enfermería
- 9556 = Higiene Oral
- 9670 = Publicidad y Marketing
- 9773 = Periodismo y Comunicación
- 9853 = Educación Básica

Los códigos restantes corresponden a otras carreras definidas en el dataset original.

---

### Nacionalidad

La variable **Nacionalidad** utiliza los códigos oficiales del conjunto de datos.

Ejemplos:

- 1 = Portuguesa
- 2 = Alemana
- 6 = Española
- 11 = Italiana
- 14 = Inglesa
- 41 = Brasileña
- 109 = Colombiana

Los demás códigos corresponden a nacionalidades registradas en el dataset original.

---

### Calificación previa

Representa el tipo de formación académica antes del ingreso a la universidad.

Ejemplos:

- 1 = Educación secundaria
- 2 = Bachillerato
- 3 = Licenciatura
- 4 = Maestría

Los demás valores siguen la codificación oficial del dataset.

---

### Calificación de la madre y del padre

- 1 = Educación secundaria
- 2 = Educación superior (Licenciatura)
- 3 = Educación superior (Maestría)
- 4 = Educación superior (Doctorado)
- 5 = Educación superior (Grado/Bachiller)
- 6 = Frecuencia de educación superior
- 9 = 12.º año de escolaridad (no completado)
- 10 = 11.º año de escolaridad
- 12 = Otro – 11.º año de escolaridad
- 14 = 10.º año de escolaridad
- 18 = Educación básica
- 19 = Educación básica (1.er ciclo)
- 22 = Educación técnica o profesional
- 27 = Educación superior – Licenciatura (1.er ciclo)
- 29 = Educación superior – Maestría (2.º ciclo)
- 30 = Educación superior – Doctorado (3.er ciclo)
- 34 = Educación superior – Técnico Superior Profesional
- 35 = Educación superior – Grado

Los demás códigos corresponden a otros niveles educativos definidos en el conjunto de datos original.

---

### Ocupación de la madre y del padre

- 0 = Otra ocupación / No especificada
- 1 = Directivos y gerentes
- 2 = Profesionales
- 3 = Técnicos y profesionales de nivel intermedio
- 4 = Personal administrativo
- 5 = Trabajadores de servicios y vendedores
- 6 = Agricultores y trabajadores agropecuarios
- 7 = Trabajadores cualificados de la industria y construcción
- 8 = Operadores de maquinaria y ensambladores
- 9 = Trabajadores no cualificados

Existen otros códigos específicos definidos por el conjunto de datos original, los cuales se mantienen sin modificación para preservar la consistencia del modelo.

---

### Importante

Esta aplicación mantiene la codificación original utilizada durante el entrenamiento del modelo de aprendizaje estadístico.

Modificar dichos códigos o reemplazarlos por valores diferentes alteraría la estructura de entrada del modelo y podría generar predicciones incorrectas.
""")
# =========================
# FORMULARIO
# =========================

st.subheader("📋 Datos principales del estudiante")

col1, col2, col3 = st.columns(3)

with col1:
    marital_status = st.number_input("Estado civil", min_value=1, max_value=6, value=1)
    application_mode = st.number_input("Modo de postulación", value=17)
    application_order = st.number_input("Orden de postulación", value=5)
    course = st.number_input("Código de carrera", value=171)
    daytime_evening = st.number_input("Tipo de asistencia", min_value=0, max_value=1, value=1)
    previous_qualification = st.number_input("Calificación previa", value=1)

with col2:
    previous_qualification_grade = st.number_input("Nota de calificación previa", value=122.0)
    admission_grade = st.number_input("Nota de admisión", value=127.3)
    gender = st.number_input("Género", min_value=0, max_value=1, value=1)
    age = st.number_input("Edad al matricularse", min_value=15, max_value=80, value=20)
    debtor = st.number_input("¿Tiene deuda?", min_value=0, max_value=1, value=0)
    tuition_fees = st.number_input("¿Pagos al día?", min_value=0, max_value=1, value=1)

with col3:
    scholarship = st.number_input("¿Tiene beca?", min_value=0, max_value=1, value=0)
    displaced = st.number_input("¿Estudiante desplazado?", min_value=0, max_value=1, value=1)
    educational_special_needs = st.number_input(
        "¿Necesidades educativas especiales?",
        min_value=0,
        max_value=1,
        value=0
    )
    international = st.number_input("¿Estudiante internacional?", min_value=0, max_value=1, value=0)
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

# =========================
# ORDEN EXACTO DE COLUMNAS
# =========================

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

# =========================
# PREDICCIÓN
# =========================

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
            "Código de carrera",
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
            marital_status,
            application_mode,
            course,
            daytime_evening,
            previous_qualification,
            nationality,
            gender,
            age,
            debtor,
            tuition_fees,
            scholarship,
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
