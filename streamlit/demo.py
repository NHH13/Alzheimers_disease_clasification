import streamlit as st
import pandas as pd
import pickle
import random

# Load the saved pipeline
@st.cache_resource
def load_pipeline(path):
    with open(path, 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

# Model path
pipeline_path = 'mi_modelo_final.pkl'
pipeline = load_pipeline(pipeline_path)

# Configure the Streamlit interface
st.image('8b6ad42c-7d27-42c8-8c72-262aac93ded9.jpg')
st.title('Binary Classification Demo for Alzheimer\'s Disease')

st.markdown("""
## Importance of Early Detection

Early detection of Alzheimer's disease is crucial for optimizing diagnosis and providing appropriate treatments. Although this classification model has been developed for educational purposes and has not been clinically validated, it can serve as a tool to better understand the factors associated with the disease.

**Note:** This classification model has been created for educational purposes and should not be used for medical diagnoses without consulting a healthcare professional.
""")

st.sidebar.header('Enter the variable values')

# Input questions based on the provided descriptions
input_values = {}

# Patient Information
input_values['PatientID'] = random.randint(4751, 6900)

# Demographic Details
input_values['Age'] = st.sidebar.slider('Age', min_value=50, max_value=90, value=70)
input_values['Gender'] = st.sidebar.selectbox('Gender', ['Male', 'Female'])
input_values['Ethnicity'] = st.sidebar.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
input_values['EducationLevel'] = st.sidebar.selectbox('Education Level', ['None', 'High School', 'Bachelor\'s', 'Higher'])

# Lifestyle Factors
input_values['BMI'] = st.sidebar.slider('BMI', min_value=15.0, max_value=40.0, value=25.0)
input_values['Smoking'] = st.sidebar.selectbox('Do you smoke?', ['No', 'Yes'])
input_values['AlcoholConsumption'] = st.sidebar.slider('Weekly alcohol consumption (units)', min_value=0, max_value=20, value=5)
input_values['PhysicalActivity'] = st.sidebar.slider('Weekly physical activity (hours)', min_value=0, max_value=10, value=5)
input_values['DietQuality'] = st.sidebar.slider('Diet quality (0-10)', min_value=0, max_value=10, value=5)
input_values['SleepQuality'] = st.sidebar.slider('Sleep quality (4-10)', min_value=4, max_value=10, value=7)

# Medical History
input_values['FamilyHistoryAlzheimers'] = st.sidebar.selectbox('Family history of Alzheimer\'s?', ['No', 'Yes'])
input_values['CardiovascularDisease'] = st.sidebar.selectbox('Cardiovascular Disease?', ['No', 'Yes'])
input_values['Diabetes'] = st.sidebar.selectbox('Diabetes?', ['No', 'Yes'])
input_values['Depression'] = st.sidebar.selectbox('Depression?', ['No', 'Yes'])
input_values['HeadInjury'] = st.sidebar.selectbox('Head Injury?', ['No', 'Yes'])
input_values['Hypertension'] = st.sidebar.selectbox('Hypertension?', ['No', 'Yes'])

# Clinical Measurements
input_values['SystolicBP'] = st.sidebar.slider('Systolic blood pressure (90-180 mmHg)', min_value=90, max_value=180, value=120)
input_values['DiastolicBP'] = st.sidebar.slider('Diastolic blood pressure (60-120 mmHg)', min_value=60, max_value=120, value=80)
input_values['CholesterolTotal'] = st.sidebar.slider('Total cholesterol (150-300 mg/dL)', min_value=150, max_value=300, value=200)
input_values['CholesterolLDL'] = st.sidebar.slider('LDL cholesterol (50-200 mg/dL)', min_value=50, max_value=200, value=100)
input_values['CholesterolHDL'] = st.sidebar.slider('HDL cholesterol (20-100 mg/dL)', min_value=20, max_value=100, value=50)
input_values['CholesterolTriglycerides'] = st.sidebar.slider('Triglycerides (50-400 mg/dL)', min_value=50, max_value=400, value=150)

# Cognitive and Functional Assessments
input_values['MMSE'] = st.sidebar.slider('MMSE Score (0-30)', min_value=0, max_value=30, value=15)
input_values['FunctionalAssessment'] = st.sidebar.slider('Functional Assessment (0-10)', min_value=0, max_value=10, value=5)
input_values['MemoryComplaints'] = st.sidebar.selectbox('Memory Complaints?', ['No', 'Yes'])
input_values['BehavioralProblems'] = st.sidebar.selectbox('Behavioral Problems?', ['No', 'Yes'])
input_values['ADL'] = st.sidebar.slider('Activities of Daily Living (0-10)', min_value=0, max_value=10, value=5)

# Symptoms
input_values['Confusion'] = st.sidebar.selectbox('Confusion?', ['No', 'Yes'])
input_values['Disorientation'] = st.sidebar.selectbox('Disorientation?', ['No', 'Yes'])
input_values['PersonalityChanges'] = st.sidebar.selectbox('Personality Changes?', ['No', 'Yes'])
input_values['DifficultyCompletingTasks'] = st.sidebar.selectbox('Difficulty completing tasks?', ['No', 'Yes'])
input_values['Forgetfulness'] = st.sidebar.selectbox('Forgetfulness?', ['No', 'Yes'])

# Confidential Information
input_values['DoctorInCharge'] = 'XXXConfid'

# Convert input values into a DataFrame
input_df = pd.DataFrame([input_values])

# Encode categorical variables
input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
input_df['Ethnicity'] = input_df['Ethnicity'].map({'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3})
input_df['EducationLevel'] = input_df['EducationLevel'].map({'None': 0, 'High School': 1, 'Bachelor\'s': 2, 'Higher': 3})
input_df['Smoking'] = input_df['Smoking'].map({'No': 0, 'Yes': 1})
input_df['FamilyHistoryAlzheimers'] = input_df['FamilyHistoryAlzheimers'].map({'No': 0, 'Yes': 1})
input_df['CardiovascularDisease'] = input_df['CardiovascularDisease'].map({'No': 0, 'Yes': 1})
input_df['Diabetes'] = input_df['Diabetes'].map({'No': 0, 'Yes': 1})
input_df['Depression'] = input_df['Depression'].map({'No': 0, 'Yes': 1})
input_df['HeadInjury'] = input_df['HeadInjury'].map({'No': 0, 'Yes': 1})
input_df['Hypertension'] = input_df['Hypertension'].map({'No': 0, 'Yes': 1})
input_df['MemoryComplaints'] = input_df['MemoryComplaints'].map({'No': 0, 'Yes': 1})
input_df['BehavioralProblems'] = input_df['BehavioralProblems'].map({'No': 0, 'Yes': 1})
input_df['Confusion'] = input_df['Confusion'].map({'No': 0, 'Yes': 1})
input_df['Disorientation'] = input_df['Disorientation'].map({'No': 0, 'Yes': 1})
input_df['PersonalityChanges'] = input_df['PersonalityChanges'].map({'No': 0, 'Yes': 1})
input_df['DifficultyCompletingTasks'] = input_df['DifficultyCompletingTasks'].map({'No': 0, 'Yes': 1})
input_df['Forgetfulness'] = input_df['Forgetfulness'].map({'No': 0, 'Yes': 1})


# Convert input values into a DataFrame
input_df = pd.DataFrame([input_values])

# Encode categorical variables
input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
input_df['Ethnicity'] = input_df['Ethnicity'].map({'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3})
input_df['EducationLevel'] = input_df['EducationLevel'].map({'None': 0, 'High School': 1, 'Bachelor\'s': 2, 'Higher': 3})
input_df['Smoking'] = input_df['Smoking'].map({'No': 0, 'Yes': 1})
input_df['FamilyHistoryAlzheimers'] = input_df['FamilyHistoryAlzheimers'].map({'No': 0, 'Yes': 1})
input_df['CardiovascularDisease'] = input_df['CardiovascularDisease'].map({'No': 0, 'Yes': 1})
input_df['Diabetes'] = input_df['Diabetes'].map({'No': 0, 'Yes': 1})
input_df['Depression'] = input_df['Depression'].map({'No': 0, 'Yes': 1})
input_df['HeadInjury'] = input_df['HeadInjury'].map({'No': 0, 'Yes': 1})
input_df['Hypertension'] = input_df['Hypertension'].map({'No': 0, 'Yes': 1})
input_df['MemoryComplaints'] = input_df['MemoryComplaints'].map({'No': 0, 'Yes': 1})
input_df['BehavioralProblems'] = input_df['BehavioralProblems'].map({'No': 0, 'Yes': 1})
input_df['Confusion'] = input_df['Confusion'].map({'No': 0, 'Yes': 1})
input_df['Disorientation'] = input_df['Disorientation'].map({'No': 0, 'Yes': 1})
input_df['PersonalityChanges'] = input_df['PersonalityChanges'].map({'No': 0, 'Yes': 1})
input_df['DifficultyCompletingTasks'] = input_df['DifficultyCompletingTasks'].map({'No': 0, 'Yes': 1})
input_df['Forgetfulness'] = input_df['Forgetfulness'].map({'No': 0, 'Yes': 1})

# Button to make the prediction
if st.sidebar.button('Make Prediction'):
    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)[:, 1]
    
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write('Diagnosis: The patient is likely diagnosed with Alzheimer\'s Disease.')
    else:
        st.write('Diagnosis: The patient is not likely diagnosed with Alzheimer\'s Disease.')
    st.write(f'Probability of Alzheimer\'s Disease: {probability[0]:.2%}')
    
    # Probability Visualization
    st.subheader('Probability Visualization:')
    prob_df = pd.DataFrame({
        'Class': ['No Alzheimer\'s', 'Alzheimer\'s'],
        'Probability': [1 - probability[0], probability[0]]
    })
    st.bar_chart(prob_df.set_index('Class'))

# Display the input values table
st.subheader('Input Values')
st.write(input_df)
