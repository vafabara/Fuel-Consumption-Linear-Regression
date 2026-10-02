import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# خواندن فایل CSV
df = pd.read_csv(r"C:\Users\sama.co\Desktop\python projects\Fuel-Consumption-Linear-Regression\FuelConsumption.csv")

# مخلوط کردن داده‌ها و تقسیم به 80 درصد Train و 20 درصد Test
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

# =========================
# رگرسیون با ENGINESIZE
# =========================

# مشخص کردن X و Y برای داده‌های Train
X_train_engine = train[["ENGINESIZE"]]
y_train_engine = train["CO2EMISSIONS"]

# مشخص کردن X و Y برای داده‌های Test
X_test_engine = test[["ENGINESIZE"]]
y_test_engine = test["CO2EMISSIONS"]

# ساخت مدل رگرسیون خطی
model_engine = LinearRegression()

# آموزش مدل
model_engine.fit(X_train_engine, y_train_engine)

# پیش‌بینی
y_pred_engine = model_engine.predict(X_test_engine)

# محاسبه R²
r2_engine = r2_score(y_test_engine, y_pred_engine)

# =========================
# رگرسیون با FUELCONSUMPTION
# =========================

X_train_fuel = train[["FUELCONSUMPTION_COMB"]]
y_train_fuel = train["CO2EMISSIONS"]

X_test_fuel = test[["FUELCONSUMPTION_COMB"]]
y_test_fuel = test["CO2EMISSIONS"]

# ساخت مدل رگرسیون خطی
model_fuel = LinearRegression()

# آموزش مدل
model_fuel.fit(X_train_fuel, y_train_fuel)

# پیش‌بینی
y_pred_fuel = model_fuel.predict(X_test_fuel)

# محاسبه R²
r2_fuel = r2_score(y_test_fuel, y_pred_fuel)

# =========================
# رسم داده‌های Train
# =========================

plt.scatter(
    train["ENGINESIZE"],
    train["CO2EMISSIONS"],
    color="blue",
    label="Engine Size - Train"
)

plt.scatter(
    train["FUELCONSUMPTION_COMB"],
    train["CO2EMISSIONS"],
    color="red",
    label="Fuel Consumption - Train"
)

# =========================
# رسم داده‌های Test
# =========================

plt.scatter(
    test["ENGINESIZE"],
    test["CO2EMISSIONS"],
    color="green",
    label="Engine Size - Test"
)

plt.scatter(
    test["FUELCONSUMPTION_COMB"],
    test["CO2EMISSIONS"],
    color="green",
    marker="x",
    label="Fuel Consumption - Test"
)

# =========================
# رسم خط‌های رگرسیون
# =========================

plt.plot(
    X_train_engine,
    model_engine.predict(X_train_engine),
    color="blue",
    linewidth=2,
    label="Engine Size Regression"
)

plt.plot(
    X_train_fuel,
    model_fuel.predict(X_train_fuel),
    color="red",
    linewidth=2,
    label="Fuel Consumption Regression"
)

# =========================
# نمایش R²
# =========================

plt.text(
    0.02,
    0.95,
    f"Engine Size R² = {r2_engine:.3f}\n"
    f"Fuel Consumption R² = {r2_fuel:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

# =========================
# تنظیمات نمودار
# =========================

plt.xlabel("Engine Size / Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.title("Linear Regression")
plt.legend()

# نمایش نمودار
plt.show()