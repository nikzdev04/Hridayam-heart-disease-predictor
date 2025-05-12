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

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("HRIDAYAM❤️ : An AI powered Heart Disease Predictor")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(['🔍 Predict', '📁 Bulk Predict', '📊 Model Information', '🩺 Features Help'])

# Tab 1: Predict
with tab1:
    st.header("Enter Patient Details")
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
        "XGBoost": "tuned_xgboost_model.pkl",
        "SVM": "SVM.pkl",
        "LogisticRegression": "LogisticRegression.pkl"
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

# Tab 2: Bulk Predict
with tab2:
    st.header("Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        if set(expected_columns).issubset(input_data.columns):
            try:
                model = pickle.load(open("tuned_xgboost_model.pkl", 'rb'))
                input_data['Prediction'] = model.predict(input_data)
                st.subheader("Predictions:")
                st.write(input_data)
                st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.warning("Please ensure the CSV has the correct columns.")

# Tab 3: Model Information
with tab3:
    model_accuracies = {
        'Decision Trees': 80.97,
        'Logistic Regression': 86.41,
        'Random Forest': 83.69,
        'Support Vector Machine': 84.22,
        'XGBoost': 89.31
    }
    df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
    fig = px.bar(df, x='Model', y='Accuracy', title="Model Accuracies", color='Accuracy', color_continuous_scale='reds')
    st.plotly_chart(fig)

# Tab 4: Feature Help
with tab4:
    st.header("🩺 Feature Help & Descriptions")

    st.markdown("""
    <style>
        .feature-box {
            background: linear-gradient(to right, #1E3C72, #2A5298);
            padding: 1.2rem;
            border-radius: 15px;
            border-left: 5px solid #FF6B6B;
            margin-bottom: 1.5rem;
            box-shadow: 2px 4px 12px rgba(0, 0, 0, 0.2);
            color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
        }

        .feature-title {
            font-weight: 600;
            font-size: 18px;
            color: #FFD700;
        }

        ul {
            padding-left: 1.2rem;
            margin-top: 0.5rem;
        }

        li {
            margin-bottom: 0.3rem;
        }
    </style>

    <div class="feature-box">
        <span class="feature-title">Age:</span> Age of the individual in years.
    </div>

    <div class="feature-box">
        <span class="feature-title">Sex:</span> Male (0), Female (1).
    </div>

    <div class="feature-box">
        <span class="feature-title">Chest Pain Type:</span>
        <ul>
            <li>0: Typical Angina – Chest pain related to decreased blood flow.</li>
            <li>1: Atypical Angina – Unusual chest pain not related to exertion.</li>
            <li>2: Non-Anginal Pain – Chest pain not related to the heart.</li>
            <li>3: Asymptomatic – No chest pain.</li>
        </ul>
    </div>

    <div class="feature-box">
        <span class="feature-title">Resting Blood Pressure:</span> Measured in mm Hg.
    </div>

    <div class="feature-box">
        <span class="feature-title">Cholesterol:</span> Serum cholesterol in mg/dL.
    </div>

    <div class="feature-box">
        <span class="feature-title">Fasting Blood Sugar:</span>
        <ul>
            <li>0: ≤ 120 mg/dL</li>
            <li>1: > 120 mg/dL</li>
        </ul>
    </div>

    <div class="feature-box">
        <span class="feature-title">Resting ECG:</span>
        <ul>
            <li>0: Normal</li>
            <li>1: ST-T Wave Abnormality</li>
            <li>2: Left Ventricular Hypertrophy</li>
        </ul>
    </div>

    <div class="feature-box">
        <span class="feature-title">MaxHR:</span> Maximum heart rate achieved during test.
    </div>

    <div class="feature-box">
        <span class="feature-title">Exercise-Induced Angina:</span>
        <ul>
            <li>0: No</li>
            <li>1: Yes</li>
        </ul>
    </div>

    <div class="feature-box">
        <span class="feature-title">Oldpeak:</span> ST depression induced by exercise compared to rest.
    </div>

    <div class="feature-box">
        <span class="feature-title">ST Slope:</span>
        <ul>
            <li>0: Upsloping – Normal</li>
            <li>1: Flat – May indicate blockage</li>
            <li>2: Downsloping – Strong indicator of heart disease</li>
        </ul>
    </div>

    <hr style="margin-top:3rem; margin-bottom:1rem; border: none; height: 1px; background-color: #FFD700;">

    <div style="text-align: center; color: white; font-family: 'Segoe UI', sans-serif;">
        <p style="font-size: 16px; font-weight: 500;">Made with ❤️ by <strong>Nikhil Kumar</strong></p>
        <div style="font-size: 24px;">
            <a href="https://www.instagram.com/nikcharanpahari.in" target="_blank" style="margin: 0 10px; text-decoration: none;">
                <img src="https://img.icons8.com/ios-filled/30/ffffff/instagram-new.png" alt="Instagram" />
            </a>
            <a href="https://www.linkedin.com/in/kumarnik12" target="_blank" style="margin: 0 10px; text-decoration: none;">
                <img src="https://img.icons8.com/ios-filled/30/ffffff/linkedin.png" alt="LinkedIn" />
            </a>
            <a href="https://github.com/nikzdev04" target="_blank" style="margin: 0 10px; text-decoration: none;">
                <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" alt="GitHub" />
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
