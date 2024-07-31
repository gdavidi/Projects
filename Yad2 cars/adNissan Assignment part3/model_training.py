import pandas as pd
import pickle
import json

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


from car_data_prep import prepare_data

# ------------------------------------------------------------ Main ------------------------------------------------------------

# Load the dataset
df = pd.read_csv('dataset.csv') 

# Preprocess the data
df_prepared = prepare_data(df)

# Split the data into features and target
X = df_prepared.drop(columns=['Price']) # Features
y = df_prepared['Price'] # Target

# Save the list of columns, to be used in the API for reordering the columns and ensuring all columns are present.
columns_list = X.columns.tolist() 
json.dump(columns_list, open("columns_list.json", "w"))  

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# Defining the model
model = ElasticNet(max_iter=15000, random_state=42) # max_iter is set to 15000 to avoid convergence warning (default is 1000) 

param_grid = {'alpha': [0.1, 1.0, 10.0],
              'l1_ratio': [0.1, 0.5, 1]}

cv = KFold(n_splits=10, shuffle=True, random_state=42) # 10-fold cross-validation

# Use GridSearchCV to search for the best hyperparameters
model_wbest_params = GridSearchCV(model, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1) # n_jobs=-1 to use all available cores
model_wbest_params.fit(X, y) # Fit the model

# Therefore, I will now set alpha and l1 ratio to the ones I found
best_model = model_wbest_params.best_estimator_ # The best model found by GridSearchCV

# Creating PKL file for the model
pickle.dump(best_model, open("trained_model.pkl", "wb"))