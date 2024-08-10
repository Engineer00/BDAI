import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the datasets
train_df = pd.read_csv('D:/BD-1/Prediction of Placement Status Data/BDAI/01 Train Data.csv')
test_df = pd.read_csv('D:/BD-1/Prediction of Placement Status Data/BDAI/02 Test Data.csv')

# Preprocess categorical features
le = LabelEncoder()
for col in ['College Name', 'Designation']:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Define features and target variables
features = ['College Name', 'Designation', 'CGPA', 'Speaking Skills', 'ML Knowledge']
target_placement = 'Placement Status'
target_graduation = 'Year of Graduation'

# Split data into training and testing sets
X_train_placement, X_test_placement, y_train_placement, y_test_placement = train_test_split(train_df[features], train_df[target_placement], test_size=0.2, random_state=42)
X_train_graduation, X_test_graduation, y_train_graduation, y_test_graduation = train_test_split(train_df[features], pd.to_numeric(train_df[target_graduation], errors='coerce'), test_size=0.2, random_state=42)

# Handle missing values separately for each split
X_train_placement.fillna(X_train_placement.mean(), inplace=True)
X_test_placement.fillna(X_test_placement.mean(), inplace=True)
X_train_graduation.fillna(X_train_graduation.mean(), inplace=True)
X_test_graduation.fillna(X_test_graduation.mean(), inplace=True)

y_train_placement.fillna(y_train_placement.mode()[0], inplace=True)
y_test_placement.fillna(y_test_placement.mode()[0], inplace=True)
y_train_graduation.fillna(y_train_graduation.median(), inplace=True)
y_test_graduation.fillna(y_test_graduation.median(), inplace=True)

# Train models
placement_model = RandomForestClassifier(n_estimators=100)
placement_model.fit(X_train_placement, y_train_placement)

graduation_model = LinearRegression()
graduation_model.fit(X_train_graduation, y_train_graduation)

# Make predictions
placement_predictions = placement_model.predict(X_test_placement)
graduation_predictions = graduation_model.predict(X_test_graduation)

# Evaluate models
placement_accuracy = accuracy_score(y_test_placement, placement_predictions)
graduation_mse = mean_squared_error(y_test_graduation, graduation_predictions)

print("Placement Prediction Accuracy:", placement_accuracy)
print("Year of Graduation Prediction MSE:", graduation_mse)

# Create DataFrames for predictions
placement_pred_df = pd.DataFrame({'Actual': y_test_placement, 'Predicted': placement_predictions})
graduation_pred_df = pd.DataFrame({'Actual': y_test_graduation, 'Predicted': graduation_predictions})

# Save predictions to CSV files
placement_pred_df.to_csv('placement_predictions.csv', index=False)
graduation_pred_df.to_csv('graduation_predictions.csv', index=False)

# Create a combined DataFrame
combined_pred_df = pd.DataFrame({
    'Actual Placement': y_test_placement,
    'Predicted Placement': placement_predictions,
    'Actual Graduation': y_test_graduation,
    'Predicted Graduation': graduation_predictions
})

# Save combined predictions to CSV file
combined_pred_df.to_csv('combined_predictions.csv', index=False)

# Evaluate models
placement_accuracy = accuracy_score(y_test_placement, placement_predictions)
graduation_mse = mean_squared_error(y_test_graduation, graduation_predictions)
print("Placement Prediction Accuracy:", placement_accuracy)
print("Year of Graduation Prediction MSE:", graduation_mse)

