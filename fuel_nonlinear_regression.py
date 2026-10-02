import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------------------
# 1) خواندن دیتاست با pandas
# ---------------------------------------------------------------
df = pd.read_csv(r"C:\Users\sama.co\Desktop\python projects\Fuel-Consumption-Linear-Regression\FuelConsumption.csv")

# x = حجم موتور (لیتر) ، y = میزان انتشار CO2 (گرم بر کیلومتر)
x = df[["ENGINESIZE"]].values      # شکل (n, 1) چون sklearn ورودی دوبعدی میخواهد
y = df["CO2EMISSIONS"].values      # شکل (n,)

# ---------------------------------------------------------------
# 2) اسکتر اولیه‌ی داده‌ها
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="steelblue", alpha=0.6, edgecolor="white")
plt.xlabel("Engine size (L)")
plt.ylabel("CO2 emissions (g/km)")
plt.title("Engine size vs CO2 emissions")
plt.grid(alpha=0.3)
plt.show()

# ---------------------------------------------------------------
# 3) جدا کردن train/test با numpy  (80% / 20%)
# ---------------------------------------------------------------
rng = np.random.default_rng(seed=42)     # seed ثابت => نتیجه قابل تکرار
indices = rng.permutation(len(df))       # ایندکس‌ها را بُر میزنیم
split = int(0.8 * len(df))               # نقطه‌ی برش 80%

train_idx = indices[:split]
test_idx = indices[split:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

# ---------------------------------------------------------------
# 4) مدل رگرسیون غیرخطی (چندجمله‌ای درجه 2) با sklearn
# ---------------------------------------------------------------
DEGREE = 2
poly = PolynomialFeatures(degree=DEGREE)
x_train_poly = poly.fit_transform(x_train)   # [1, x, x^2]  (فقط روی train فیت میشود)
x_test_poly = poly.transform(x_test)         # روی test فقط transform

model = LinearRegression()
model.fit(x_train_poly, y_train)

# ---------------------------------------------------------------
# 5) ارزیابی
# ---------------------------------------------------------------
r2_train = r2_score(y_train, model.predict(x_train_poly))
r2_test = r2_score(y_test, model.predict(x_test_poly))
print(f"Degree: {DEGREE}")
print(f"R2 (train): {r2_train:.3f}")
print(f"R2 (test) : {r2_test:.3f}")

# ---------------------------------------------------------------
# 6) نمودار نهایی: train، test، منحنی مدل و R2
# ---------------------------------------------------------------
x_line = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
y_line = model.predict(poly.transform(x_line))

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="steelblue", alpha=0.5, label="Train (80%)")
plt.scatter(x_test, y_test, color="orange", alpha=0.8, label="Test (20%)")
plt.plot(x_line, y_line, color="crimson", linewidth=2.5, label=f"Polynomial fit (deg={DEGREE})")

plt.xlabel("Engine size (L)")
plt.ylabel("CO2 emissions (g/km)")
plt.title("Non-linear regression: Engine size vs CO2 emissions")
plt.grid(alpha=0.3)
plt.legend(loc="lower right")

# نمایش R2 در گوشه‌ی نمودار (مختصات نسبی محور: 0 تا 1)
plt.text(
    0.03, 0.95,
    f"R² train = {r2_train:.3f}\nR² test  = {r2_test:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)

plt.tight_layout()
plt.savefig("regression_plot.png", dpi=150)
plt.show()
