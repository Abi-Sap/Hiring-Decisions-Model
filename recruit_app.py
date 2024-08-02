import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load('random_model.pkl')

def main():
    categorical_features = {
        'Gender': ['Male', 'Female'],
        'EducationLevel': ['Wassce', 'Bachelors Degree', 'Masters', 'PHD'],
        'RecruitmentStrategy': ['Aggressive', 'Moderate', 'Conservative'],
    }
    
    # Streamlit app configuration
    st.set_page_config(
        page_title="Hiring Decision Prediction",
        page_icon="hiring-logo.png",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Adding custom CSS to set the background color
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FEFBBD;
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Recruitment Decision Prediction')
    
    # Load and display the image using PIL
    image = Image.open('hiring-img.jpg')
    st.image(image, use_column_width=True)
    
    # Sidebar input fields
    name = st.sidebar.text_input('Name')
    email = st.sidebar.text_input('Email')
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    gender = st.sidebar.selectbox('Gender', options=[0, 1], format_func=lambda x: categorical_features['Gender'][x])
    education_level = st.sidebar.selectbox('Education Level', options=[1, 2, 3, 4], format_func=lambda x: categorical_features['EducationLevel'][x-1])
    experience_years = st.sidebar.number_input('Experience Years', min_value=0, max_value=15, value=5)
    previous_companies = st.sidebar.number_input('Previous Companies', min_value=1, max_value=5, value=2)
    distance_from_company = st.sidebar.number_input('Distance From Company', min_value=0.0, max_value=100.0, value=10.0)
    interview_score = st.sidebar.number_input('Interview Score', min_value=0, max_value=100, value=50)
    skill_score = st.sidebar.number_input('Skill Score', min_value=0, max_value=100, value=50)
    personality_score = st.sidebar.number_input('Personality Score', min_value=0, max_value=100, value=50)
    recruitment_strategy = st.sidebar.selectbox('Recruitment Strategy', options=[1, 2, 3], format_func=lambda x: categorical_features['RecruitmentStrategy'][x-1])
    
    # Predict button
    if st.sidebar.button('Predict'):
        input_data = np.array([[age, gender, education_level, experience_years, previous_companies,
                                distance_from_company, interview_score, skill_score, personality_score,
                                recruitment_strategy]])
        prediction = model.predict(input_data)
        
        if prediction == 1:
            st.success(f"{name} is likely to be hired.")
        else:
            st.error(f"{name} is unlikely to be hired.")

if __name__ == "__main__":
    main()
