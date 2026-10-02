import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


# =========================
# خواندن فایل CSV
# =========================

df = pd.read_csv(
    r"C:\Users\sama.co\Desktop\python projects\Fuel-Consumption-Linear-Regression\FuelConsumption.csv"
)


# =========================
# انتخاب Features و Target
# =========================

features = [
    "ENGINESIZE",
    "CYLINDERS",
    "TRANSMISSION",
    "FUELTYPE",
    "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB",
    "FUELCONSUMPTION_COMB_MPG"
]

X = df[features]
y = df["CO2EMISSIONS"]


# =========================
# تبدیل داده‌های متنی به عدد
# =========================

X = pd.get_dummies(
    X,
    columns=["TRANSMISSION", "FUELTYPE"],
    dtype=int
)


# =========================
# مخلوط کردن داده‌ها
# =========================

df_model = pd.concat([X, y], axis=1)

df_model = df_model.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================
# تقسیم Train و Test
# =========================

split = int(len(df_model) * 0.8)

train = df_model.iloc[:split]
test = df_model.iloc[split:]


X_train = train.drop("CO2EMISSIONS", axis=1)
y_train = train["CO2EMISSIONS"]

X_test = test.drop("CO2EMISSIONS", axis=1)
y_test = test["CO2EMISSIONS"]


# =========================
# ساخت مدل Multiple Linear Regression
# =========================

regr = LinearRegression()

regr.fit(X_train, y_train)


# =========================
# پیش‌بینی
# =========================

y_hat = regr.predict(X_test)

x = np.asarray(X_test)
y = np.asarray(y_test)


# =========================
# Mean Squared Error
# =========================

print(
    "Mean Squared Error: %.2f"
    % np.mean((y_hat - y) ** 2)
)


# =========================
# Explained variance / R²
# 1 = Perfect Prediction
# =========================

print(
    "Variance score: %.2f"
    % regr.score(x, y)
)