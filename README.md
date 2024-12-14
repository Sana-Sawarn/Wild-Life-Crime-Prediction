# Wild-Life-Crime-Prediction
"Developed a wildlife crime prediction system using a dataset (2017-2022) to assess district-level risk. Users can select a year (2015-2030), state, and district to receive a high-risk or low-risk prediction for targeted conservation and crime prevention efforts."

# Data Source & Preprocessing:
- **Dataset:** District-wise wildlife crime data (2017-2022) with socio-economic indicators.
- **Preprocessing:**
  - Removed redundant and missing columns.
  - Handled missing values using mean/mode imputation.
  - One-hot encoded categorical variables (state and district names).
  - Standardized numerical features using **StandardScaler**.

# Modeling:
- **Algorithm:** Random Forest Classifier.
- **Imbalance Handling:** SMOTE for oversampling.
- **Hyperparameter Tuning:** GridSearchCV.
- **Risk Categorization:** Districts in the top 20% of crime counts are "High Risk," others are "Low Risk."

# User Interaction & Output:
- Users input the year (2015-2030), state, and district.
- The app processes the input, scales data, and predicts the wildlife crime risk level ("High Risk" or "Low Risk").
- Output is displayed on an interactive **Streamlit dashboard**.


![Screenshot 2024-12-14 160527](https://github.com/user-attachments/assets/8a84e6bc-f5ca-41c5-8acd-a61bac421cec)





# Deployment:
- Model, scaler, and label encoder saved using **joblib**.
- Interactive **Streamlit app** for live predictions.

# Tech Stack:
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn
- **Deployment:** Streamlit for interactive user interface
- **Tools:** Joblib for model persistence

# Outcome:
A user-friendly system that allows users to predict district-level wildlife crime risk for any year, aiding targeted conservation and crime prevention efforts.
