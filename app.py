import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('districtwise_wildlife_crime_rate_2017_onwards.csv')

# Identify columns to drop (those starting with 'Unnamed')
columns_to_drop = [col for col in data.columns if 'Unnamed' in col]
data_cleaned = data.drop(columns=columns_to_drop)

# Fill missing values in 'Wildlife_Protection' with 0
data_cleaned['Wildlife_Protection'] = data_cleaned['Wildlife_Protection'].fillna(0)

# One-hot encode 'state_name' and 'district_name'
data_encoded = pd.get_dummies(data_cleaned, columns=['state_name', 'district_name'], drop_first=True)

# Compute total crimes per district
crime_counts = data.groupby(['state_name', 'district_name']).size().reset_index(name='crime_count')

# Define high-risk as districts in the top 20% of crime counts
limit = crime_counts['crime_count'].quantile(0.1)
crime_counts['risk_level'] = crime_counts['crime_count'].apply(lambda x: 'High Risk' if x >= limit else 'Low Risk')

# Merge back with socio-economic indicators or other features
wildlife_crime = pd.read_csv('districtwise_wildlife_crime_rate_2017_onwards.csv')  # Replace with actual path
district_data = pd.merge(crime_counts, wildlife_crime, on=['state_name', 'district_name'], how='left')

# Handle missing values
numerical_cols = ['crime_count']  # Add other numerical columns if present
for col in numerical_cols:
    district_data[col] = district_data[col].fillna(district_data[col].mean())

categorical_cols = ['state_name', 'district_name']  # Adjust column names as per your DataFrame
for col in categorical_cols:
    district_data[col] = district_data[col].fillna(district_data[col].mode()[0])

# Drop columns that are completely empty or have only NaN values
district_data = district_data.dropna(axis=1, how='all')

columns_to_drop = ['id', 'year', 'state_code', 'district_code', 'registration_circles']  # Add any other unwanted columns
district_data = district_data.drop(columns=columns_to_drop, errors='ignore')

# Define feature columns and target
feature_cols = ['crime_count']  # Add more as needed
X = district_data[feature_cols]
y = district_data['risk_level']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Standardize the feature(s)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Address class imbalance with SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Save feature names for later use
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest and GridSearchCV
crime_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(crime_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model with balanced data
grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best model from Grid Search
best_crime_classifier = grid_search.best_estimator_

# Predict on the test set
y_pred = best_crime_classifier.predict(X_test_scaled)

# Classification metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model, scaler, and label encoder
joblib.dump(best_crime_classifier, 'best_crime_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model, scaler, label encoder, and feature names saved.")

import streamlit as st

# Load the model, scaler, label encoder, and feature names
model = joblib.load('best_crime_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')  # Load feature names

# Streamlit app title
st.title("Wildlife Crime Risk Prediction")

# Sidebar inputs for year, state, and district
st.sidebar.header("Select Input Parameters")
year = st.sidebar.selectbox("Select Year", list(range(2015, 2031)))
state = st.sidebar.text_input("Enter State Name", value="ExampleState")
district = st.sidebar.text_input("Enter District Name", value="ExampleDistrict")

# Display selected input values for confirmation
st.write("### Input Values:")
st.write("**Year:**", year)
st.write("**State:**", state)
st.write("**District:**", district)

# When the "Predict" button is pressed
if st.button("Predict Risk Level"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'state_name': [state],
        'district_name': [district],
        'year': [year]
    })

    # One-hot encode the input data to match the model's features
    input_data = pd.get_dummies(input_data, columns=['state_name', 'district_name'], drop_first=True)

    # Add missing columns to match the model's expected input
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the model’s feature order
    input_data = input_data[feature_names]

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the loaded model
    prediction_encoded = model.predict(input_data_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # Display the prediction
    st.write("### Prediction Result:")
    st.write(f"The predicted wildlife crime risk level is: **{prediction_label}**")
