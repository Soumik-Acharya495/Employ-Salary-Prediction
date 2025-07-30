# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset (replace with your actual dataset)
# For this example, we'll create a synthetic dataset
def create_synthetic_data():
    np.random.seed(42)
    num_samples = 1000
    
    # Generate synthetic features
    experience = np.random.randint(0, 20, num_samples)
    age = experience + np.random.randint(22, 30, num_samples)
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_samples)
    job_title = np.random.choice(['Developer', 'Manager', 'Data Scientist', 'Analyst', 'Engineer'], num_samples)
    company_size = np.random.choice(['Small', 'Medium', 'Large'], num_samples)
    
    # Generate salary based on features with some noise
    salary = (
        30000 + 
        experience * 2500 + 
        (education_level == 'Bachelor') * 10000 + 
        (education_level == 'Master') * 20000 + 
        (education_level == 'PhD') * 30000 + 
        (job_title == 'Manager') * 15000 + 
        (job_title == 'Data Scientist') * 20000 + 
        (company_size == 'Medium') * 5000 + 
        (company_size == 'Large') * 10000 + 
        np.random.normal(0, 5000, num_samples)
    )
    
    data = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Education': education_level,
        'JobTitle': job_title,
        'CompanySize': company_size,
        'Salary': salary
    })
    
    return data

# Load or create dataset
try:
    df = pd.read_csv('salary_data.csv')  # Replace with your dataset
    print("Loaded dataset from CSV file")
except FileNotFoundError:
    df = create_synthetic_data()
    print("Created synthetic dataset for demonstration")

# Exploratory Data Analysis
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nDescriptive Statistics:")
print(df.describe())

# Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(x='Education', y='Salary', data=df)
plt.title('Salary Distribution by Education Level')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Experience', y='Salary', hue='JobTitle', data=df)
plt.title('Salary vs Experience by Job Title')
plt.show()

# Data Preprocessing
# Handle categorical variables
label_encoders = {}
categorical_cols = ['Education', 'JobTitle', 'CompanySize']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature and target separation
X = df.drop('Salary', axis=1)
y = df['Salary']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title(f'{name} - Actual vs Predicted Salary')
    plt.show()

# Display results
print("\nModel Performance Comparison:")
results_df = pd.DataFrame(results).T
print(results_df)

# Feature Importance (for Random Forest)
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Save the best model and preprocessing objects
model_data = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': X.columns.tolist()
}

with open('salary_predictor.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Prediction function
def predict_salary(input_data):
    # Load the saved model
    with open('salary_predictor.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    features = model_data['features']
    
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    
    # Ensure all features are present and in correct order
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing features with default value
    
    input_df = input_df[features]  # Reorder columns
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction[0]

# Example usage of the prediction function
example_input = {
    'Age': 32,
    'Experience': 5,
    'Education': 'Master',
    'JobTitle': 'Data Scientist',
    'CompanySize': 'Large'
}

predicted_salary = predict_salary(example_input)
print(f"\nPredicted Salary for the input: ${predicted_salary:,.2f}")
