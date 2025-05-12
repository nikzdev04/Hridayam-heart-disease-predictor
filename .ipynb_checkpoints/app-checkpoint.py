import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
    return href

st.title("Heart Disease Prediction")

tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<=120 mg/dL", ">120 mg/dL"])
    resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == ">120 mg/dL" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    models = {
        "XGBoost": "xgboost_model.pkl",
        "SVM": "svm_model.pkl",
        "GridSearch RF": "grid_rf_model.pkl"
    }

    loaded_models = {}
    for name, path in models.items():
        try:
            loaded_models[name] = pickle.load(open(path, 'rb'))
        except Exception as e:
            st.error(f"Failed to load {name}: {e}")

    if st.button("Submit"):
        predictions = {}
        for model_name, model in loaded_models.items():
            prediction = model.predict(input_data)[0]
            predictions[model_name] = prediction

        st.subheader("Individual Model Predictions:")
        for model_name, pred in predictions.items():
            result = "No heart disease" if pred == 0 else "Heart disease"
            st.write(f"**{model_name}**: {result}")

        votes = list(predictions.values())
        final_result = max(set(votes), key=votes.count)
        st.subheader("Majority Vote Result:")
        if final_result == 0:
            st.success("Final Verdict: No heart disease detected.")
        else:
            st.error("Final Verdict: Heart disease detected.")

with tab2:
    st.title("Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        if set(expected_columns).issubset(input_data.columns):
            try:
                model = pickle.load(open("xgboost_model.pkl", 'rb'))
                input_data['Prediction'] = model.predict(input_data)
                st.subheader("Predictions:")
                st.write(input_data)
                st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.warning("Please ensure the CSV has the correct columns.")

with tab3:
    model_accuracies = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 88.50,
        'Grid Search RF': 89.75,
        'XGBoost': 89.31
    }
    df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
    fig = px.bar(df, x='Model', y='Accuracy', title="Model Accuracies")
    st.plotly_chart(fig)
