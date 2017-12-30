import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

CSV_PATH = "abalone.data.txt"

# No missing values, "sex" is only categorical attribute
abalone = pd.read_csv(CSV_PATH)

# No difference between M/F so we group M and F together and separate from I
sex_to_int = {"I":0, "M":1, "F":1}
abalone["sex"].replace(to_replace=sex_to_int, inplace=True)

# Age = Rings + 1.5
# We can get remove rings attribute from dataframe since new label is age 
abalone["age"] = abalone["rings"] + 1.5
abalone.drop("rings", axis=1, inplace=True)

X = np.array(abalone.drop("age", axis=1)) # Numerical data
y = np.array(abalone["age"]) # Label (age)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# May add more transformations, perhaps experimenting with different attribute combinations
pipeline = Pipeline([
    ("std_scaler", StandardScaler()) # feature scaling
    ])

X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.fit_transform(X_test)

# Fine-tuning
          

hyperparameters = {"n_estimators":[10, 15, 30],
                        "max_features":[2, 3, 4, 5],
                        "max_depth":[5, 10, 15, 20, 30],
                        "bootstrap":[True, False],
                        "warm_start":[True, False]}

##rand_forest_reg = RandomForestRegressor()
##grid_search = GridSearchCV(rand_forest_reg, hyperparameters, cv=10,
##                           scoring="neg_mean_squared_error")
##
##grid_search.fit(X_train_preprocessed, y_train)
##
##model = grid_search.best_estimator_
##
##joblib.dump(model, "abalone_model.pkl")

model = joblib.load("abalone_model.pkl")

predictions = model.predict(X_test_preprocessed)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = model.score(X_test_preprocessed, y_test)
r = np.sqrt(r2)
print("Correlation:", r)

print("Predictions:", predictions[:50])
print("Labels:", y_test[:50])
