# Movie Rating Prediction - Regression Project
# Predicts average movie rating using RandomForestRegressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# 1. Load dataset
data = pd.read_csv("movie_ratings_data.csv")

# 2. Feature scaling - normalization
ss = StandardScaler()
data["budget"] =  ss.fit_transform(data[["budget"]])

# 3. Separate features and target
X = data.iloc[:,:-1]
y = data['average_rating']

# 4. Encode categorical variable 'genre' 

le = LabelEncoder()
X["genre"] = le.fit_transform(X["genre"])



# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 6. Build and train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# score
print(rf.score(X_test,y_test)*100)

# 7. Predict on test set
y_pred = rf.predict(X_test)

# 8. Evaluation

mae = mean_absolute_error(y_test, y_pred)