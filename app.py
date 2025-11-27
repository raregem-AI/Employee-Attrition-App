
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Streamlit application title
st.title('Employee Attrition Prediction App')

# Introduction and purpose of the application
st.markdown("""
This application predicts the likelihood of employee attrition based on various input features.
It utilizes a trained XGBoost model and applies the same preprocessing steps used during training.
""")

st.write("Streamlit app.py file generated successfully.")

# Load the trained model and preprocessing objects
best_xgb_model = joblib.load('best_xgb_model.joblib')
SELECTED_FEATURES = joblib.load('selected_features.joblib')
ORDINAL_CAT_ORDER = joblib.load('ordinal_cat_order.joblib')
ohe_transformers = joblib.load('ohe_transformers.joblib')
scaler_params = joblib.load('scaler_params.joblib')

st.write("Model and preprocessing objects loaded successfully!")

# Re-define original categorical features for OHE as in training notebook
original_ohe_cat_features = [
    'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime',
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'WorkLifeBalance'
]

# Re-define CONTINUOUS_FEATURES as it was in training notebook before scaling
CONTINUOUS_FEATURES_TRAINING = [
    'Age', 'BusinessTravel', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'IncomePerAge', 'AgeRisk', 'HourlyRateRisk', 'DistanceRisk',
    'ShortCompanyTenure', 'NumCompaniesAdjusted', 'AverageCompanyTenure',
    'JobHopperIndicator', 'AttritionRiskScore'
]

def preprocess_input(input_data):
    """
    Applies all necessary preprocessing steps to raw user input.
    """
    # 1. Convert input to DataFrame (handling single row input)
    input_df = pd.DataFrame([input_data])

    # Make a copy to avoid SettingWithCopyWarning
    processed_df = input_df.copy()

    # 2. Apply feature engineering steps (from cell mrAbnoMOghVI in notebook)
    processed_df['IncomePerAge'] = processed_df['MonthlyIncome'] / processed_df['Age']
    processed_df["AgeRisk"] = (processed_df["Age"] < 34).astype(int)
    processed_df["HourlyRateRisk"] = (processed_df["HourlyRate"] < 60).astype(int)
    processed_df["DistanceRisk"] = (processed_df["DistanceFromHome"] >= 20).astype(int)
    processed_df["ShortCompanyTenure"] = (processed_df["YearsAtCompany"] < 4).astype(int)
    processed_df['NumCompaniesAdjusted'] = processed_df['NumCompaniesWorked'].replace(0, 1)
    processed_df['AverageCompanyTenure'] = processed_df["TotalWorkingYears"] / processed_df["NumCompaniesAdjusted"]
    processed_df['JobHopperIndicator'] = ((processed_df["NumCompaniesAdjusted"] > 2) & (processed_df["AverageCompanyTenure"] < 2.0)).astype(int)
    processed_df["AttritionRiskScore"] = processed_df["AgeRisk"] + processed_df["HourlyRateRisk"] + processed_df["DistanceRisk"] + processed_df["ShortCompanyTenure"] + processed_df['JobHopperIndicator']

    # Adjust 'Education' and 'JobLevel' values (if they exist and apply)
    # These were specific corrections, so apply them if the values match
    processed_df.loc[processed_df['Education'] == 15, 'Education'] = 5
    processed_df.loc[processed_df['JobLevel'] == 7, 'JobLevel'] = 5

    # 3. Apply ordinal encoding to 'BusinessTravel' (from cell JBcq-pYFgnks)
    # Ensure 'BusinessTravel' column exists and is of object type for mapping
    if 'BusinessTravel' in processed_df.columns:
        temp_business_travel = processed_df['BusinessTravel'].astype(str)
        for idx, value in enumerate(ORDINAL_CAT_ORDER):
            temp_business_travel.loc[temp_business_travel == value] = idx
        processed_df['BusinessTravel'] = temp_business_travel.astype(int)

    # 4. Apply One-Hot Encoding (from cell QtHdIYhCh6eV)
    for feature in original_ohe_cat_features:
        if feature in processed_df.columns:
            ohe = ohe_transformers[feature]
            feature_df = pd.DataFrame(processed_df[feature])

            # Get column names for the encoded features
            # ohe.categories_[0] contains all unique categories found during fit
            # drop='first' means the first category is dropped, so we start from the second
            new_encoded_columns = [f"{feature}_{val}_ohe" for val in ohe.categories_[0][1:]]

            encoded_df = pd.DataFrame(ohe.transform(feature_df), columns=new_encoded_columns, index=processed_df.index, dtype=int)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            processed_df.drop(columns=[feature], inplace=True) # Drop original categorical feature

    # 5. Apply Standard Scaling to CONTINUOUS_FEATURES_TRAINING (from cell gOQVmGM8ic3r)
    for cont_feature in CONTINUOUS_FEATURES_TRAINING:
        if cont_feature in processed_df.columns and cont_feature in scaler_params:
            mean_value = scaler_params[cont_feature]['mean']
            std_dev = scaler_params[cont_feature]['std']
            if std_dev != 0: # Avoid division by zero for constant features
                processed_df[cont_feature] = (processed_df[cont_feature] - mean_value) / std_dev
            else:
                processed_df[cont_feature] = 0.0 # Or some other appropriate handling for constant features
        elif cont_feature not in processed_df.columns: # Handle cases where a feature might be missing in input but expected for scaling
            processed_df[cont_feature] = 0.0 # Fill with 0 or mean/median of training data scaled (0 for scaled)

    # 6. Ensure final DataFrame contains only SELECTED_FEATURES in the correct order
    # Any features in SELECTED_FEATURES not present will be added with 0 (important for OHE categories not in input)
    final_features = pd.DataFrame(columns=SELECTED_FEATURES)
    processed_df = pd.concat([final_features, processed_df], ignore_index=True, sort=False)
    processed_df = processed_df[SELECTED_FEATURES].fillna(0) # Fill any NaNs that might result from concat or missing OHE columns

    return processed_df

st.write("Preprocessing functions defined successfully!")

st.subheader('Enter Employee Details:')

user_input = {}

# Grouping inputs into columns for better readability
col1, col2, col3 = st.columns(3)

with col1:
    user_input['Age'] = st.slider('Age', min_value=18, max_value=60, value=35)
    user_input['DailyRate'] = st.slider('Daily Rate', min_value=100, max_value=1500, value=800)
    user_input['DistanceFromHome'] = st.slider('Distance From Home', min_value=1, max_value=29, value=10)
    user_input['HourlyRate'] = st.slider('Hourly Rate', min_value=30, max_value=100, value=65)
    user_input['MonthlyIncome'] = st.slider('Monthly Income', min_value=1000, max_value=20000, value=6500)
    user_input['MonthlyRate'] = st.slider('Monthly Rate', min_value=2000, max_value=27000, value=14000)
    user_input['NumCompaniesWorked'] = st.slider('Number of Companies Worked', min_value=0, max_value=9, value=2)
    user_input['PercentSalaryHike'] = st.slider('Percent Salary Hike', min_value=11, max_value=25, value=14)
    user_input['TotalWorkingYears'] = st.slider('Total Working Years', min_value=0, max_value=40, value=10)
    user_input['TrainingTimesLastYear'] = st.slider('Training Times Last Year', min_value=0, max_value=6, value=3)

with col2:
    user_input['YearsAtCompany'] = st.slider('Years At Company', min_value=0, max_value=40, value=5)
    user_input['YearsInCurrentRole'] = st.slider('Years In Current Role', min_value=0, max_value=18, value=3)
    user_input['YearsSinceLastPromotion'] = st.slider('Years Since Last Promotion', min_value=0, max_value=15, value=1)
    user_input['YearsWithCurrManager'] = st.slider('Years With Current Manager', min_value=0, max_value=17, value=3)

    # Categorical features
    user_input['BusinessTravel'] = st.selectbox('Business Travel', options=ORDINAL_CAT_ORDER, index=1)
    user_input['Department'] = st.selectbox('Department', options=ohe_transformers['Department'].categories_[0], index=1)
    user_input['Education'] = st.selectbox('Education', options=ohe_transformers['Education'].categories_[0], index=2)
    user_input['EducationField'] = st.selectbox('Education Field', options=ohe_transformers['EducationField'].categories_[0], index=2)
    user_input['EnvironmentSatisfaction'] = st.selectbox('Environment Satisfaction', options=ohe_transformers['EnvironmentSatisfaction'].categories_[0], index=2)
    user_input['Gender'] = st.selectbox('Gender', options=ohe_transformers['Gender'].categories_[0], index=0)

with col3:
    user_input['JobInvolvement'] = st.selectbox('Job Involvement', options=ohe_transformers['JobInvolvement'].categories_[0], index=2)
    user_input['JobLevel'] = st.selectbox('Job Level', options=ohe_transformers['JobLevel'].categories_[0], index=0)
    user_input['JobRole'] = st.selectbox('Job Role', options=ohe_transformers['JobRole'].categories_[0], index=7)
    user_input['JobSatisfaction'] = st.selectbox('Job Satisfaction', options=ohe_transformers['JobSatisfaction'].categories_[0], index=2)
    user_input['MaritalStatus'] = st.selectbox('Marital Status', options=ohe_transformers['MaritalStatus'].categories_[0], index=1)
    user_input['OverTime'] = st.selectbox('Over Time', options=ohe_transformers['OverTime'].categories_[0], index=0)
    user_input['PerformanceRating'] = st.selectbox('Performance Rating', options=ohe_transformers['PerformanceRating'].categories_[0], index=0)
    user_input['RelationshipSatisfaction'] = st.selectbox('Relationship Satisfaction', options=ohe_transformers['RelationshipSatisfaction'].categories_[0], index=2)
    user_input['StockOptionLevel'] = st.selectbox('Stock Option Level', options=ohe_transformers['StockOptionLevel'].categories_[0], index=0)
    user_input['WorkLifeBalance'] = st.selectbox('Work Life Balance', options=ohe_transformers['WorkLifeBalance'].categories_[0], index=2)

st.write("User input interface created successfully!")


# Prediction button
if st.button('Predict Attrition'):
    # Preprocess the user input
    processed_input = preprocess_input(user_input)

    # Make prediction
    prediction_proba = best_xgb_model.predict_proba(processed_input.values)[:, 1]
    attrition_probability = prediction_proba[0]

    # Display results
    st.subheader('Prediction Results:')
    if attrition_probability > 0.5:
        st.error(f'High probability of Attrition: {attrition_probability:.2f}')
        st.write("This employee is likely to leave the company.")
    else:
        st.success(f'Low probability of Attrition: {attrition_probability:.2f}')
        st.write("This employee is likely to stay with the company.")

st.write("Prediction logic added successfully!")

# Load the trained model and preprocessing objects
best_xgb_model = joblib.load('best_xgb_model.joblib')
SELECTED_FEATURES = joblib.load('selected_features.joblib')
ORDINAL_CAT_ORDER = joblib.load('ordinal_cat_order.joblib')
ohe_transformers = joblib.load('ohe_transformers.joblib')
scaler_params = joblib.load('scaler_params.joblib')

st.write("Model and preprocessing objects loaded successfully!")
