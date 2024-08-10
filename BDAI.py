import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the datasets
train_df = pd.read_csv('01 Train Data.csv')
test_df = pd.read_csv('02 Test Data.csv')

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

# Make predictions
placement_predictions = placement_model.predict(test_df[features])

# Create a new DataFrame with the original columns and the predicted placement
output_df = test_df.copy()
output_df['Predicted Placement'] = placement_predictions

# Save the output DataFrame to a CSV file
output_df.to_csv('output.csv', index=False)