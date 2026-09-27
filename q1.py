import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\ML\assignments\assignment4\train.csv")
data = df[["GrLivArea", "SalePrice"]].dropna()
X = data[["GrLivArea"]]
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

area = float(input("\nEnter house area in sq. ft: "))
new_house = pd.DataFrame({"GrLivArea":[area]})
predicted_price = model.predict(new_house)
print("Predicted House Price: $", round(predicted_price[0], 2))
plt.scatter(X_test, y_test, label="Actual Prices")
plt.plot(X_test, y_pred, label="Regression Line")

plt.xlabel("House Area (sq. ft)")
plt.ylabel("House Price ($)")
plt.title("House Price Prediction Based on House Area")
plt.legend()
plt.show()