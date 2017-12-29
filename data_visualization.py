import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "abalone.data.txt"

abalone = pd.read_csv(CSV_PATH)

# Many histograms are tail heavy
# We will try transforming these attributes to have more bell-shaped distributions
abalone.hist(bins=50, figsize=(10,8))
plt.show()

abalone["age"] = abalone["rings"] + 1.5
abalone.drop("rings", axis=1, inplace=True)


plt.show()
 
abalone.plot(kind="scatter", x="diameter", y="length", alpha=0.5,
             figsize=(8,5), c="age", cmap=plt.get_cmap("jet"),
             colorbar=True, title="Effect of Shell Size on Age")
plt.show()

# Shell diameter and length appear to have a positive, moderate, linear relationship with age

abalone.plot(kind="scatter", x="height", y="age", alpha=0.5,
             title="Abalone Height and Age")
plt.show()

# Height has a positive, steeper, moderate, linear relationship

abalone.plot(kind="scatter", x="whole_weight", y="age", alpha=0.5,
             title="Abalone Weight and Age")
plt.show()

# Whole weight and other weight attributes appear to have a logarithmic
# relationship with age

abalone.plot(kind="scatter", x="whole_weight", y="age", alpha=0.5,
             logx=True, title="Log of Weight and Age")
plt.show()

# Scaling weight with logx creates a more linear relationship

corr_matrix = abalone.corr()
print(corr_matrix["age"].sort_values(ascending=False))

attributes = ["age", "diameter", "whole_weight", "length", "height"]
scatter_matrix(abalone[attributes], figsize=(10, 6))
plt.show()

abalone["ln(whole_weight)"] = np.log(abalone["whole_weight"])

abalone["log10(whole_weight)"] = np.log10(abalone["whole_weight"])

abalone["log2(whole_weight)"] = np.log2(abalone["whole_weight"])

print()
corr_matrix = abalone.corr()
print(corr_matrix["age"].sort_values(ascending=False))

