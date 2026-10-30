import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

##### Linear regression (predict a number)
# CSV file: Salary_Data.csv  (columns: YearsExperience, Salary)

df = pd.read_csv("Salary_Data.csv")   # load the CSV file as a table
print(df.head())                      # first 5 rows
print(df.columns)                     # column names
print(df.isnull().sum())              # count missing values in each column

X = df[["YearsExperience"]]           # inputs (double brackets = table)
y = df["Salary"]                      # answer (single brackets = one column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

r2 = r2_score(y_test, y_pred)
print("r2_score: ", r2)

##### Logistic regression and KNN (predict a type)
# CSV file: diabetes.csv  (last column: Outcome, 0 = no diabetes, 1 = diabetes)

df2 = pd.read_csv("diabetes.csv")
print(df2.head())
print(df2.columns)
print(df2.isnull().sum())

X = df2.drop("Outcome", axis=1)       # all columns except Outcome
y = df2["Outcome"]                    # the answer: 0 or 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

model_K = KNeighborsClassifier(n_neighbors=3)
model_K.fit(X_train, y_train)

pred = model.predict(X_test)
print(pred)

pred_K = model_K.predict(X_test)
print(pred_K)

print("Accuracy_score: ", accuracy_score(y_test, pred))
print("Accuracy_score: ", accuracy_score(y_test, pred_K))