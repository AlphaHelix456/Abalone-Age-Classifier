import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

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
    X, y, test_size=0.2, random_state=42)

# May add more transformations, perhaps experimenting with different attribute combinations
pipeline = Pipeline([
    ("std_scaler", StandardScaler()) # feature scaling
    ])

X_train_preprocessed = pipeline.fit_transform(X_train)

# Fine-tuning

def fine_tune(clf, hyperparameters, scoring="neg_mean_squared_error", cv=10):
    grid_search = GridSearchCV(clf, hyperparameters, cv=cv, scoring=scoring)
    grid_search.fit(X_train_preprocessed, y_train)
    print(grid_search.best_params_)
    print(grid_search.score(X_train_preprocessed, y_train))
          
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
rand_forest_reg = RandomForestRegressor()
svr = SVR()


lin_hyperparameters = {"fit_intercept":[True, False],
                       "normalize":[True, False],
                       "copy_X":[True, False]}

tree_hyperparameters = {"max_depth":[1, 5, 10, 30, None],
                        "max_features":[2, 4, 6, None, "sqrt"],
                        "presort":[True, False]}

rand_forest_hyperparameters = {"n_estimators":[10, 15, 30],
                        "max_features":[2, 3, 4, 5],
                        "max_depth":[5, 10, 15, 20, 30, None],
                        "bootstrap":[True, False],
                        "warm_start":[True, False]}

svr_hyperparameters = {"kernel":["rbf", "poly", "sigmoid", "linear"],
                       "shrinking":[True, False]}

fine_tune(lin_reg, lin_hyperparameters, scoring="r2")
fine_tune(tree_reg, tree_hyperparameters, scoring="r2")
fine_tune(rand_forest_reg, rand_forest_hyperparameters, scoring="r2")
fine_tune(svr, svr_hyperparameters, scoring="r2")

# Linear regression, decision tree regression, and svr all perform about the
# same with the grid search
# Random forest regressor is the best model




