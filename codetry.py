'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file (adjust the path to your file)
df = pd.read_excel('trialdata.xlsx')

# 1. Gender Distribution - Pie Chart
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Gender Distribution of Respondents')
plt.ylabel('')  # Hide the y-axis label
plt.show()

# 2. Screen Time vs Stress Frequency - Scatter Plot
df['Screen Time'] = df['What is your average daily screen time (including work and leisure)?'].apply(
    lambda x: {'Less than 2 hours': 1, '2-4 hours': 2, '4-6 hours': 3, 'More than 6 hours': 4}.get(x, 0))

df['Stress Frequency'] = df['How often do you feel stressed in a week?'].apply(
    lambda x: {'Never': 0, '1-2 days': 1, '3-4 days': 2, '5-7 days': 3}.get(x, 0))

plt.figure(figsize=(10, 6))
plt.scatter(df['Screen Time'], df['Stress Frequency'], color='b', alpha=0.5)
plt.title('Screen Time vs Stress Frequency')
plt.xlabel('Average Screen Time (in hours)')
plt.ylabel('Stress Frequency (1-7 days)')
plt.grid(True)
plt.show()

# 3. Coping Mechanism Distribution - Bar Chart
plt.figure(figsize=(10, 6))
coping_mechanisms = df['What is your primary coping mechanism for stress?'].value_counts()
coping_mechanisms.plot(kind='bar', color='lightcoral')
plt.title('Primary Coping Mechanism for Stress')
plt.ylabel('Count of Responses')
plt.xticks(rotation=45, ha='right')
plt.show()

# 4. Stress Causes by Gender - Grouped Bar Chart
plt.figure(figsize=(10, 6))
stress_causes_by_gender = pd.crosstab(df['Gender'], df['What is the primary cause of your stress?'])
stress_causes_by_gender.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Stress Causes by Gender')
plt.xlabel('Gender')
plt.ylabel('Count of Responses')
plt.xticks(rotation=0)
plt.legend(title='Stress Causes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. Coping Mechanism by Activity Level - Grouped Bar Chart
plt.figure(figsize=(12, 6))
coping_mechanism_by_activity = pd.crosstab(df['How do you rate your daily physical activity level?'], df['What is your primary coping mechanism for stress?'])
coping_mechanism_by_activity.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Coping Mechanism by Physical Activity Level')
plt.xlabel('Physical Activity Level')
plt.ylabel('Count of Responses')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Coping Mechanism', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6. Awareness of Wearable Devices - Pie Chart
plt.figure(figsize=(8, 6))
wearable_awareness = df['Are you aware of wearable devices that track stress (e.g. smartwatches)?'].value_counts()
wearable_awareness.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ffcc99', '#66b3ff'])
plt.title('Awareness of Wearable Devices that Track Stress')
plt.ylabel('')  # Hide the y-axis label
plt.show()

# 7. Trust in Machine Learning for Mental Health Suggestions - Bar Chart
plt.figure(figsize=(10, 6))
trust_ml = df['Would you trust a machine learning model to provide mental health suggestions?'].value_counts()
trust_ml.plot(kind='bar', color='lightgreen')
plt.title('Trust in Machine Learning for Mental Health Suggestions')
plt.ylabel('Count of Responses')
plt.xticks(rotation=0)
plt.show()

# 8. Correlation Heatmap of Numeric Data
plt.figure(figsize=(10, 8))
numeric_data = df[['Screen Time', 'Stress Frequency']]
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Screen Time and Stress Frequency')
plt.show()

# 9. Coping Mechanisms by Stress Frequency - Grouped Bar Chart
plt.figure(figsize=(12, 6))
coping_by_stress = pd.crosstab(df['How often do you feel stressed in a week?'], df['What is your primary coping mechanism for stress?'])
coping_by_stress.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Coping Mechanism by Stress Frequency')
plt.xlabel('Stress Frequency')
plt.ylabel('Count of Responses')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset from your Excel file
data = pd.read_excel('trialdata.xlsx')  # Replace with the correct file path

# Step 2: Drop irrelevant non-numeric columns (e.g., 'Name', 'Timestamp')
data = data.drop(columns=['Timestamp', 'Name'])

# Step 3: Handle specific columns with non-numeric strings
# Mapping the 'How often do you feel stressed in a week?' column to numeric values
stress_map = {
    'Never': 0,
    '1-2 days': 1,
    '3-4 days': 2,
    '5-6 days': 3,
    'Every day': 4
}
data['How often do you feel stressed in a week?'] = data['How often do you feel stressed in a week?'].map(stress_map)

# Mapping 'What is your average daily screen time?' to numeric values
screen_time_map = {
    '0-1 hours': 0,
    '1-2 hours': 1,
    '2-3 hours': 2,
    '3-4 hours': 3,
    '4-6 hours': 4,
    '6+ hours': 5
}
data['What is your average daily screen time (including work and leisure)?'] = data['What is your average daily screen time (including work and leisure)?'].map(screen_time_map)

# Step 4: Apply Label Encoding on other categorical columns
categorical_columns = [
    'Gender',
    'What is the primary cause of your stress?',
    'How do you usually react to stress?',
    'Do you experience sleep disturbances due to stress?',
    'What is your primary coping mechanism for stress?',
    'How do you rate your daily physical activity level?',
    'How frequently do you socialize or interact with friends/family?',
    'Are you aware of wearable devices that track stress (e.g. smartwatches)?',
    'Do you think machine learning can predict mental stress effectively?',
    'Would you trust a machine learning model to provide mental health suggestions?',
    'What data do you think is most relevant for predicting stress?',
    'How comfortable are you with sharing personal data (e.g. wearable data) for research purposes?'
]

# Initialize LabelEncoder
le = LabelEncoder()

# Encode all categorical columns
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Step 5: Handle missing values using SimpleImputer (e.g., filling with the median)
imputer = SimpleImputer(strategy='median')  # You can also use 'mean' or 'most_frequent'
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Step 6: Define Features (X) and Target (y)
X = data_imputed.drop(columns=['How often do you feel stressed in a week?'])  # Features
y = data_imputed['How often do you feel stressed in a week?']  # Target variable

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling (if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Apply a Regression Model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Make Predictions
y_pred = model.predict(X_test)

# Step 11: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (MSE)
print(f'Mean Squared Error: {mse}')

# Optional: Print Model Coefficients for interpretation
print(f'Model Coefficients: {model.coef_}')
print(f'Model Intercept: {model.intercept_}')

# --- Visualization Section ---

# 1. Plot Actual vs Predicted values (for model evaluation)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# 2. Plot Residuals (for model evaluation)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.residplot(y_pred, residuals, lowess=True, color='green')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
'''



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset from your Excel file
data = pd.read_excel('trialdata.xlsx')  # Replace with your file path

# Step 2: Drop irrelevant columns (e.g., 'Name', 'Timestamp')
data = data.drop(columns=['Timestamp', 'Name'])

# Step 3: Map specific non-numeric columns to numeric
# Mapping 'How often do you feel stressed in a week?' to numeric values
stress_map = {
    'Never': 0,
    '1-2 days': 1,
    '3-4 days': 2,
    '5-6 days': 3,
    'Every day': 4
}
data['How often do you feel stressed in a week?'] = data['How often do you feel stressed in a week?'].map(stress_map)

# Mapping 'What is your average daily screen time?' to numeric values
screen_time_map = {
    '0-1 hours': 0,
    '1-2 hours': 1,
    '2-3 hours': 2,
    '3-4 hours': 3,
    '4-6 hours': 4,
    '6+ hours': 5
}
data['What is your average daily screen time (including work and leisure)?'] = data['What is your average daily screen time (including work and leisure)?'].map(screen_time_map)

# Step 4: Label Encoding for categorical columns
categorical_columns = [
    'Gender',
    'What is the primary cause of your stress?',
    'How do you usually react to stress?',
    'Do you experience sleep disturbances due to stress?',
    'What is your primary coping mechanism for stress?',
    'How do you rate your daily physical activity level?',
    'How frequently do you socialize or interact with friends/family?',
    'Are you aware of wearable devices that track stress (e.g. smartwatches)?',
    'Do you think machine learning can predict mental stress effectively?',
    'Would you trust a machine learning model to provide mental health suggestions?',
    'What data do you think is most relevant for predicting stress?',
    'How comfortable are you with sharing personal data (e.g. wearable data) for research purposes?'
]

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Step 5: Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='median')  # Use 'mean', 'most_frequent' if needed
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Step 6: Define Features (X) and Target (y)
X = data_imputed.drop(columns=['How often do you feel stressed in a week?'])  # Features
y = data_imputed['How often do you feel stressed in a week?']  # Target variable

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Applying Different Models ---

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# 2. Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)

# 3. Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# 4. Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)

# 5. Support Vector Regressor (SVR)
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)

# 6. K-Nearest Neighbors Regressor (KNN)
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)

# Store MSE for comparison
model_mse = {
    'Linear Regression': mse_lr,
    'Decision Tree': mse_dt,
    'Random Forest': mse_rf,
    'Gradient Boosting': mse_gb,
    'Support Vector Regressor': mse_svr,
    'K-Nearest Neighbors': mse_knn
}

# Sort models by their Mean Squared Errors
sorted_models = sorted(model_mse.items(), key=lambda x: x[1])

# Print the sorted models
print("Model Comparison (Sorted by MSE):")
for model, mse in sorted_models:
    print(f'{model}: {mse}')

# --- Visualization Section ---
# Plot Actual vs Predicted for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted - Linear Regression')
plt.show()

# Plot Residuals for Linear Regression
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_lr, y=residuals_lr, lowess=True, color='green')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Linear Regression')
plt.show()
