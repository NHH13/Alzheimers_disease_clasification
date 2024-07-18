import streamlit as st
import pandas as pd
import pickle
from preprocessor import FlexiblePreprocessor, apply_transformations  # Importar la clase y la función necesarias

# Load the saved pipeline
def load_pipeline(path):
    with open(path, 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

# Define the function to make predictions
def predict(pipeline, data):
    prediction = pipeline.predict(data)
    probability = pipeline.predict_proba(data)[:, 1]
    return prediction, probability

# Model path
pipeline_path = r'C:\Users\LENOVO\OneDrive\Documentos\GitHub\Alzheimers_disease_clasification\models\pipeline_model.pkl'
pipeline = load_pipeline(pipeline_path)

# Configure the Streamlit interface
st.image(r"C:\Users\LENOVO\OneDrive\Documentos\GitHub\Alzheimers_disease_clasification\images\8b6ad42c-7d27-42c8-8c72-262aac93ded9.jpg")  # Replace with the path to your image
st.title('Binary Classification Demo for Alzheimer\'s Disease')

st.markdown("""
## Importance of Early Detection

Early detection of Alzheimer's disease is crucial for optimizing diagnosis and providing appropriate treatments. Although this classification model has been developed for educational purposes and has not been clinically validated, it can serve as a tool to better understand the factors associated with the disease.

**Note:** This classification model has been created for educational purposes and should not be used for medical diagnoses without consulting a healthcare professional.
""")

st.sidebar.header('Enter the variable values')

# Input questions based on the provided description
input_values = {}

# Cognitive and Functional Assessments
input_values['FunctionalAssessment'] = st.sidebar.slider('Functional Assessment (0-10)', min_value=0, max_value=10, value=5)
input_values['ADL'] = st.sidebar.slider('Activities of Daily Living (0-10)', min_value=0, max_value=10, value=5)
input_values['MMSE'] = st.sidebar.slider('MMSE Score (0-30)', min_value=0, max_value=30, value=15)
input_values['MemoryComplaints'] = st.sidebar.selectbox('Memory Complaints?', ['No', 'Yes'])
input_values['BehavioralProblems'] = st.sidebar.selectbox('Behavioral Problems?', ['No', 'Yes'])

# Lifestyle Factors
weight = st.sidebar.number_input('Weight in kg', min_value=30, max_value=150, value=70)
height = st.sidebar.number_input('Height in cm', min_value=120, max_value=220, value=170)
input_values['BMI'] = weight / (height / 100) ** 2
input_values['Smoking'] = st.sidebar.selectbox('Do you smoke?', ['No', 'Yes'])
input_values['AlcoholConsumption'] = st.sidebar.slider('Weekly alcohol consumption (units)', min_value=0, max_value=20, value=5)
input_values['PhysicalActivity'] = st.sidebar.slider('Weekly physical activity (hours)', min_value=0, max_value=10, value=5)
input_values['DietQuality'] = st.sidebar.slider('Diet quality (0-10)', min_value=0, max_value=10, value=5)
input_values['SleepQuality'] = st.sidebar.slider('Sleep quality (4-10)', min_value=4, max_value=10, value=7)

# Medical History
input_values['Hypertension'] = st.sidebar.selectbox('Hypertension?', ['No', 'Yes'])
input_values['Diabetes'] = st.sidebar.selectbox('Diabetes?', ['No', 'Yes'])
input_values['FamilyHistoryAlzheimers'] = st.sidebar.selectbox('Family history of Alzheimer\'s?', ['No', 'Yes'])
input_values['Depression'] = st.sidebar.selectbox('Depression?', ['No', 'Yes'])
input_values['CardiovascularDisease'] = st.sidebar.selectbox('Cardiovascular Disease?', ['No', 'Yes'])
input_values['HeadInjury'] = st.sidebar.selectbox('Head Injury?', ['No', 'Yes'])

# Clinical Measurements
input_values['SystolicBP'] = st.sidebar.slider('Systolic blood pressure (90-180 mmHg)', min_value=90, max_value=180, value=120)
input_values['DiastolicBP'] = st.sidebar.slider('Diastolic blood pressure (60-120 mmHg)', min_value=60, max_value=120, value=80)
input_values['CholesterolTotal'] = st.sidebar.slider('Total cholesterol (150-300 mg/dL)', min_value=150, max_value=300, value=200)
input_values['CholesterolLDL'] = st.sidebar.slider('LDL cholesterol (50-200 mg/dL)', min_value=50, max_value=200, value=100)
input_values['CholesterolHDL'] = st.sidebar.slider('HDL cholesterol (20-100 mg/dL)', min_value=20, max_value=100, value=50)
input_values['CholesterolTriglycerides'] = st.sidebar.slider('Triglycerides (50-400 mg/dL)', min_value=50, max_value=400, value=150)

# Symptoms
input_values['Forgetfulness'] = st.sidebar.selectbox('Forgetfulness?', ['No', 'Yes'])
input_values['Disorientation'] = st.sidebar.selectbox('Disorientation?', ['No', 'Yes'])
input_values['DifficultyCompletingTasks'] = st.sidebar.selectbox('Difficulty completing tasks?', ['No', 'Yes'])
input_values['Confusion'] = st.sidebar.selectbox('Confusion?', ['No', 'Yes'])

# Demographic Details
input_values['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
input_values['Ethnicity'] = st.sidebar.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
input_values['EducationLevel'] = st.sidebar.selectbox('Education Level', ['None', 'High School', 'Bachelor\'s', 'Higher'])
input_values['Age'] = st.sidebar.slider('Age', min_value=60, max_value=90, value=70)

# Convert input values into a DataFrame
input_df = pd.DataFrame([input_values])

# Convert categorical variables to numerical values
input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
input_df['Ethnicity'] = input_df['Ethnicity'].map({'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3})
input_df['EducationLevel'] = input_df['EducationLevel'].map({'None': 0, 'High School': 1, 'Bachelor\'s': 2, 'Higher': 3})
input_df['Smoking'] = input_df['Smoking'].map({'No': 0, 'Yes': 1})
input_df['FamilyHistoryAlzheimers'] = input_df['FamilyHistoryAlzheimers'].map({'No': 0, 'Yes': 1})
input_df['Hypertension'] = input_df['Hypertension'].map({'No': 0, 'Yes': 1})
input_df['Diabetes'] = input_df['Diabetes'].map({'No': 0, 'Yes': 1})
input_df['Depression'] = input_df['Depression'].map({'No': 0, 'Yes': 1})
input_df['MemoryComplaints'] = input_df['MemoryComplaints'].map({'No': 0, 'Yes': 1})
input_df['BehavioralProblems'] = input_df['BehavioralProblems'].map({'No': 0, 'Yes': 1})
input_df['Forgetfulness'] = input_df['Forgetfulness'].map({'No': 0, 'Yes': 1})
input_df['Disorientation'] = input_df['Disorientation'].map({'No': 0, 'Yes': 1})
input_df['DifficultyCompletingTasks'] = input_df['DifficultyCompletingTasks'].map({'No': 0, 'Yes': 1})
input_df['Confusion'] = input_df['Confusion'].map({'No': 0, 'Yes': 1})
input_df['CardiovascularDisease'] = input_df['CardiovascularDisease'].map({'No': 0, 'Yes': 1})
input_df['HeadInjury'] = input_df['HeadInjury'].map({'No': 0, 'Yes': 1})

# Apply the transformations
input_df = apply_transformations(input_df)

# Ensure the input_df has the same columns as the model expects
expected_features = ['FunctionalAssessment', 'ADL', 'MMSE', 'MemoryComplaints',
                     'BehavioralProblems', 'SleepQuality', 'CholesterolHDL',
                     'CholesterolTriglycerides', 'PhysicalActivity', 'DietQuality',
                     'CholesterolTotal', 'CholesterolLDL', 'AlcoholConsumption', 'BMI',
                     'SystolicBP', 'DiastolicBP', 'EducationLevel', 'AgeGroup', 'HealthRisk',
                     'Ethnicity', 'Smoking', 'Gender', 'Hypertension', 'Diabetes',
                     'Forgetfulness', 'FamilyHistoryAlzheimers', 'Depression',
                     'DifficultyCompletingTasks', 'Disorientation', 'Confusion']

input_df = input_df[expected_features]

# Debugging: Print input dataframe to ensure correct values
print("Input DataFrame for prediction:")
print(input_df)

import plotly.express as px

# Button to make the prediction
if st.sidebar.button('Make Prediction'):
    prediction, probability = predict(pipeline, input_df)
    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.markdown('<div style="color:red; font-size:24px;">Diagnosis: The patient is likely diagnosed with Alzheimer\'s Disease.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:green; font-size:24px;">Diagnosis: The patient is not likely diagnosed with Alzheimer\'s Disease.</div>', unsafe_allow_html=True)
    
    st.write(f'Probability of High Risk: {probability[0]:.2%}')
    
    # Create a histogram using plotly
    fig = px.bar(
        x=['No Alzheimer', 'Alzheimer'],
        y=[1 - probability[0], probability[0]],
        labels={'x': 'Diagnosis', 'y': 'Probability'},
        title='Probability of Diagnosis'
    )
    fig.update_layout(yaxis_tickformat='.2%', title_x=0.5)
    st.plotly_chart(fig)

