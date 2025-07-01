from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib



def feature_engineering(df):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # 1. Convert boolean-like text columns ('yes'/'no') to numerical (1/0)
    bool_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for col in bool_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # 2. Apply Label Encoding to the 'furnishingstatus' column if it exists
    if "furnishingstatus" in df.columns:
        le = LabelEncoder()
        df["furnishingstatus_encoded"] = le.fit_transform(df["furnishingstatus"])
        df.drop("furnishingstatus", axis=1, inplace=True)  # Drop the original text column

    # 3. Drop rows with any missing values (NaN) for data cleanliness
    df.dropna(inplace=True)

    # 4. Reset the DataFrame index after dropping rows to maintain sequential indexing
    df.reset_index(drop=True, inplace=True)

    return df




df=pd.read_csv("Housing.csv")
df=feature_engineering(df)
print(df.head(5))
print(df.isnull())

df=df[(df['area']>500) & (df['area']<15000)]
df=df[(df['price']>500000) & (df['price']<15000000)]


print('shape',df.shape)

print(df.head(5))



sns.histplot(df['price'],color='blue')
plt.title('Histogram of the price')
plt.show()

sns.scatterplot(x='area',y='price',data=df,color='pink')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area VS Price')
plt.show()

cor=df.corr()
sns.heatmap(cor,annot=True,fmt='.2f',cmap='coolwarm')
plt.show()





# prepare data
x = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model,"linear_model.pkl")


y_pred = model.predict(X_test)

# eval
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
sns.scatterplot(x="Actual", y="Predicted", data=results, color="blue")
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

results.to_csv("predictions.csv",index=False)