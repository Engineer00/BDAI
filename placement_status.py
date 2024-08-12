import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv('01 Train Data.csv')
test_df = pd.read_csv('02 Test Data.csv')

# Handle missing values
numeric_cols = ['CGPA', 'Speaking Skills', 'ML Knowledge']
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())
train_df = train_df.fillna('')
test_df = test_df.fillna('')

# Preprocess categorical features
le_college_name = LabelEncoder()
le_designation = LabelEncoder()
le_placement_status = LabelEncoder()

train_df['College Name'] = le_college_name.fit_transform(train_df['College Name'])
test_df['College Name'] = le_college_name.transform(test_df['College Name'])
train_df['Designation'] = le_designation.fit_transform(train_df['Designation'])
test_df['Designation'] = le_designation.transform(test_df['Designation'])

# Combine 'Placement Status' columns
placement_statusCombined = pd.concat([train_df['Placement Status'], test_df['Placement Status']])

# Handle NaN in 'Placement Status'
placement_statusCombined = placement_statusCombined.fillna('Unknown')

# Fit LabelEncoder on combined column
le_placement_status.fit(placement_statusCombined)

# Transform 'Placement Status' columns
train_df['Placement Status'] = le_placement_status.transform(train_df['Placement Status'])
test_df['Placement Status'] = le_placement_status.transform(test_df['Placement Status'])

# Convert 'Year of Graduation' to numeric
train_df['Year of Graduation'] = pd.to_numeric(train_df['Year of Graduation'], errors='coerce')
test_df['Year of Graduation'] = pd.to_numeric(test_df['Year of Graduation'], errors='coerce')

# Create new feature 'Years Since Graduation'
train_df['Years Since Graduation'] = 2024 - train_df['Year of Graduation']
test_df['Years Since Graduation'] = 2024 - test_df['Year of Graduation']

# Define features and target variables
features = ['College Name', 'Designation', 'CGPA', 'Speaking Skills', 'ML Knowledge', 'Years Since Graduation']
target_placement = 'Placement Status'

# Split data into training and testing sets
X_train_placement, X_test_placement, y_train_placement, y_test_placement = train_test_split(train_df[features], train_df[target_placement], test_size=0.2, random_state=42, stratify=train_df[target_placement])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_placement, y_train_placement)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train_placement, y_train_placement)

# Make predictions
placement_predictions = best_model.predict(test_df[features])

# Transform predictions to show "not placed" or "placed"
placement_predictions = ['placed' if x == 1 else 'not placed' for x in placement_predictions]

# Save the predictions to a CSV file
output_df = test_df.copy()
output_df['Predicted Placement'] = placement_predictions
output_df.to_csv('placement_predictions.csv', index=False)

# Print the classification report
print("Classification Report:")
print(classification_report(test_df[target_placement], best_model.predict(test_df[features]), zero_division=0))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(test_df[target_placement], best_model.predict(test_df[features])))

# Feature importance
feature_importance = best_model.feature_importance_
print("Feature Importance:")
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.2f}")
    
accuracy = accuracy_score(test_df[target_placement], best_model.predict(test_df[features]))
print("Accuracy:", accuracy)
accuracy_percentage = accuracy * 100
print("Accuracy (%):", accuracy_percentage)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# Save the plot to a file
plt.savefig('feature_importance_plot.png')

print("Code execution completed successfully!")