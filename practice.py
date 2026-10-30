from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

diabetes = load_diabetes()
print(diabetes.data[:5])
print(diabetes.target[:5])

X=diabetes.data
y=diabetes.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_pred[:10])

mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)

r2_score = r2_score(y_test,y_pred)
print("r2_score: ",r2_score)

#####Logistic regression

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_wine()
X=data.data
y=data.target

#Binary classification
X=X[y!=2]
y=y[y!=2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

model_K=KNeighborsClassifier(n_neighbors=3)
model_K.fit(X_train,y_train)

pred = model.predict(X_test)
print(pred)

pred_K=model_K.predict(X_test)
print(pred_K)

print("Accuracy_score: ",accuracy_score(y_test,pred))
print("Accuracy_score: ",accuracy_score(y_test,pred_K))