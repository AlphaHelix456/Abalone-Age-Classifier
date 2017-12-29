import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

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

# Training and evaluation linear regression model on training set
lin_reg = LinearRegression()
lin_reg.fit(X_train_preprocessed, y_train)
print("Linear Regression Evaluation")
print("Predictions:", lin_reg.predict(X_train[:30]))
print("Labels:", y_train[:30])
print("Accuracy:", lin_reg.score(X_train_preprocessed, y_train))

predictions = lin_reg.predict(X_train_preprocessed)
lin_mse = mean_squared_error(y_train, predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE:", lin_rmse)

# Typical prediction error on training set is 2.187 years
# Model is underfitting the data

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_preprocessed, y_train)
predictions = tree_reg.predict(X_train_preprocessed)
tree_mse = mean_squared_error(y_train, predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree Evaluation")
print("Accuracy:", tree_reg.score(X_train_preprocessed, y_train))
print("RMSE:", tree_rmse)

# Model is overfitting the data so much so that the accuracy is 100%

# 10-fold cross-validation to better evaluate Decision Tree model
tree_scores = cross_val_score(tree_reg, X_train_preprocessed, y_train,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

def display_scores(scores):
    print("Scores from CV:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
# RMSE is higher than that of linear regression meaning the decision tree model
# performs worse. This confirms that the decision tree model is overfitting the data
