import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# DATA : https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv

data = pd.read_csv("real_estate_data.csv")

#print(data.head())

# Correcting data

#print(data.isna().sum())
data.dropna(inplace=True)
#print(data.isna().sum())

# splitting data

X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

regression_tree = DecisionTreeRegressor(criterion = "friedman_mse")
regression_tree.fit(X_train, Y_train)

print(regression_tree.score(X_test, Y_test))

prediction = regression_tree.predict(X_test)
print("$",(prediction - Y_test).abs().mean()*1000)