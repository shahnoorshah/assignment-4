import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv(r"C:\Users\Lenovo\Desktop\ML\assignments\assignment4\train.csv")
data=df[["GrLivArea","BedroomAbvGr","SalePrice"]].dropna()
X=data[["GrLivArea","BedroomAbvGr"]]
y=data["SalePrice"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Coefficients: ")
print("House Area Coefficient: ",model.coef_[0])
print("Bedroom Coefficeint: ", model.coef_[1])
print("\nIntercept: ",model.intercept_)
print("\nModel Performance: ")
print("MAE: ",mean_absolute_error(y_test,y_pred))
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE: ",rmse)
print("R2 Score: ",r2_score(y_test,y_pred))
area=float(input("\nEnter house area in sq. ft: "))
bedrooms=int(input("Enter number of bedrooms: "))
new_house=pd.DataFrame({
    "GrLivArea":[area],
    "BedroomAbvGr":[bedrooms]
})
predicted_price=model.predict(new_house)
print("\nPredicted House Price: $",round(predicted_price[0],2))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()]
)
plt.show()