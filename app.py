import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- 1. Configuración de la página ---
st.set_page_config(page_title="Sistema de Alerta Temprana", page_icon="🎓", layout="centered")

# --- 2. Carga de Modelos y Datos (Caché para mayor velocidad) ---
@st.cache_resource
def load_assets():
    # Cargar el preprocesador y el modelo
    preprocessor = joblib.load('preprocesador_desercion.pkl')
    model = load_model('modelo_mlp_desercion.h5')
    
    # Cargar el dataset para obtener las columnas y valores por defecto (medianas)
    df = pd.read_csv('students_dropout_academic_success.csv')
    X_cols = df.drop('target', axis=1).columns
    # Calculamos la mediana de cada columna numérica para usarla como valor por defecto
    default_values = {col: df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0] for col in X_cols}
    
    return preprocessor, model, X_cols, default_values

preprocessor, model, X_cols, default_values = load_assets()

# --- 3. Interfaz de Usuario (Frontend) ---
st.title("🎓 Sistema de Alerta Temprana de Deserción")
st.write("Ingrese los datos del estudiante para evaluar su riesgo de deserción durante el primer año académico. [cite: 1, 2]")

st.markdown("### 📊 Datos Clave del Estudiante")

# Crear columnas para organizar mejor los inputs
col1, col2 = st.columns(2)

with col1:
    tuition_up_to_date = st.selectbox("¿Matrícula al día?", ["Sí", "No"])
    debtor = st.selectbox("¿Es deudor?", ["No", "Sí"])
    scholarship = st.selectbox("¿Es becado?", ["No", "Sí"])
    gender = st.selectbox("Género", ["Femenino", "Masculino"])
    age = st.number_input("Edad al matricularse", min_value=15, max_value=80, value=18)

with col2:
    cu_1st_sem_app = st.number_input("Unidades aprobadas (1er Semestre)", min_value=0, max_value=20, value=5)
    cu_1st_sem_grade = st.number_input("Nota promedio (1er Semestre)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    cu_2nd_sem_app = st.number_input("Unidades aprobadas (2do Semestre)", min_value=0, max_value=20, value=5)
    cu_2nd_sem_grade = st.number_input("Nota promedio (2do Semestre)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)

# --- 4. Lógica de Predicción (Backend) ---
if st.button("Evaluar Riesgo Estudiantil", type="primary"):
    
    # Preparamos un diccionario con todos los valores por defecto
    student_data = default_values.copy()
    
    # Sobreescribimos solo las variables que el usuario modificó en la web
    student_data['Tuition fees up to date'] = 1 if tuition_up_to_date == "Sí" else 0
    student_data['Debtor'] = 1 if debtor == "Sí" else 0
    student_data['Scholarship holder'] = 1 if scholarship == "Sí" else 0
    student_data['Gender'] = 1 if gender == "Masculino" else 0
    student_data['Age at enrollment'] = age
    student_data['Curricular units 1st sem (approved)'] = cu_1st_sem_app
    student_data['Curricular units 1st sem (grade)'] = cu_1st_sem_grade
    student_data['Curricular units 2nd sem (approved)'] = cu_2nd_sem_app
    student_data['Curricular units 2nd sem (grade)'] = cu_2nd_sem_grade

    # Convertimos a DataFrame para que el preprocesador lo entienda
    input_df = pd.DataFrame([student_data], columns=X_cols)
    
    try:
        # Preprocesar y predecir
        processed_input = preprocessor.transform(input_df)
        prob = model.predict(processed_input, verbose=0)[0][0]
        
        # --- 5. Protocolo de Intervención (Fase 6) --- [cite: 41, 42, 43, 44]
        st.markdown("---")
        st.markdown("### 📋 Resultados de la Evaluación")
        
        # Umbral del 70% como lo pide el documento
        if prob > 0.7:
            st.error(f"**ESTADO: ALTO RIESGO DE DESERCIÓN ({prob * 100:.1f}%)**")
            st.warning("**⚠️ Protocolo de Bienestar Activado:**\n"
                       "1. Generación de alerta automática al tutor asignado.\n"
                       "2. Envío de invitación formal a tutoría académica.\n"
                       "3. Contacto por parte de asistencia social / psicológica.")
        elif prob > 0.4:
            st.warning(f"**ESTADO: Riesgo Moderado ({prob * 100:.1f}%)**")
            st.info("Recomendación: Realizar seguimiento preventivo al finalizar el semestre.")
        else:
            st.success(f"**ESTADO: Riesgo Bajo ({prob * 100:.1f}%)**")
            st.write("El estudiante presenta indicadores académicos y socioeconómicos estables.")
            
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")