import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


df = pd.read_csv(r"C:\Users\Lenovo\Desktop\ML\assignments\assignment4\train.csv")
data = df[["GrLivArea","BedroomAbvGr","SalePrice"]].dropna()
X = data[["GrLivArea","BedroomAbvGr"]]
y = data["SalePrice"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
linear_model=LinearRegression()
linear_model.fit(X_train, y_train)
linear_prediction=linear_model.predict(X_test)
linear_r2=r2_score(y_test, linear_prediction)
print("Standard Linear Regression")
print("R2 Score: ",linear_r2)
poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_prediction = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_prediction)
print("\nPolynomial Regression")
print("R2 Score:", poly_r2)
print("\nComparison of R2 Scores:")
print("Linear Regression:", linear_r2)
print("Polynomial Regression:", poly_r2)
if poly_r2 > linear_r2:
    print("\nPolynomial Regression has a higher R2 score.")
elif poly_r2 < linear_r2:
    print("\nLinear Regression has a higher R2 score.")
else:
    print("\nBoth models have the same R2 score.")