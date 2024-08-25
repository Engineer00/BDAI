import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
data = pd.read_csv('Final Lead Data.csv')

# Convert the 'Created' column to datetime
data['Created'] = pd.to_datetime(data['Created'])

# Extract the year from the 'Created' column
data['Created Year'] = data['Created'].dt.year

# Handle missing values in the Academic Year column
# Here, we'll drop rows with missing Academic Year values
data = data.dropna(subset=['Academic Year'])

# Preprocess categorical features
data['New College Name'] = LabelEncoder().fit_transform(data['New College Name'])
data['Branch/ Specialisation'] = LabelEncoder().fit_transform(data['Branch/ Specialisation'])

# Define the features (X) and target (y)
X = data[['New College Name', 'Academic Year', 'Branch/ Specialisation']]

# Calculate the graduation year
data['Forecasted Graduation Year'] = data['Created Year'] + (4 - data['Academic Year'])

# Print the first few rows to verify
print(data[['Academic Year', 'Created Year', 'Forecasted Graduation Year']])

# Define y as the calculated graduation year with dtype int
y = data['Forecasted Graduation Year'].astype(int)

# Remove NaN values from y
y = y.dropna()

# Remove corresponding rows from X
X = X.loc[y.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5]
}

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Use the best model to make predictions and round to nearest whole number
y_pred = grid_search.best_estimator_.predict(X_test).round().astype(int)

# Evaluate the model using various metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-Squared (R2): {r2:.2f}')

# Use the trained model to forecast graduation year for all data and round to nearest whole number
forecasted_graduation_year = grid_search.best_estimator_.predict(X).round()

# Fill NaN values with 0 and convert to integers
forecasted_graduation_year = np.nan_to_num(forecasted_graduation_year, nan=0).astype(int)

# Create a new DataFrame with all columns and the predicted graduation year
output_df = data.copy()
output_df['Forecasted Graduation Year'] = pd.Series(forecasted_graduation_year).reindex(output_df.index).fillna(0).astype(int)

# Identify rows with 0 values in Forecasted Graduation Year
zero_value_rows = output_df[output_df['Forecasted Graduation Year'] == 0]

# Calculate actual graduation year for these rows
for index, row in zero_value_rows.iterrows():
    academic_year = row['Academic Year']
    created_year = row['Created Year']
    actual_graduation_year = created_year + (5 - academic_year)
    output_df.loc[index, 'Forecasted Graduation Year'] = actual_graduation_year

# Verify the changes
print(output_df['Forecasted Graduation Year'])

# Write the output DataFrame to a CSV file
output_df.to_csv('predicted_graduation_year.csv', index=False)
