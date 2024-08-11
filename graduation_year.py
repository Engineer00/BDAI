
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# Load the dataset
data = pd.read_csv('Final Lead Data.csv')

# Preprocess the data
data['New College Name'] = LabelEncoder().fit_transform(data['New College Name'])
data['Branch/ Specialisation'] = LabelEncoder().fit_transform(data['Branch/ Specialisation'])

# Define the features (X) and target (y)
X = data[['New College Name', 'Academic Year', 'Branch/ Specialisation']]
y = data['Academic Year'] + 4

# Remove NaN values from y
y = y.dropna()

# Remove corresponding rows from X
X = X.loc[y.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],  # Change 'auto' to 'sqrt'
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5]
}
# Initialize the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use the best model to make predictions
y_pred = np.round(grid_search.best_estimator_.predict(X_test))

# Evaluate the model using various metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-Squared (R2): {r2:.2f}')

# Use the trained model to forecast graduation year for all data
forecasted_graduation_year = np.round(grid_search.best_estimator_.predict(X))

# Create a new DataFrame with all columns and the predicted graduation year
output_df = data.copy()
output_df['Forecasted Graduation Year'] = pd.Series(forecasted_graduation_year).reindex(output_df.index)

# Write the output DataFrame to a CSV file
output_df.to_csv('predicted_graduation_year.csv', index=False)