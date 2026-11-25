from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Week 09 ile birebir ayni veri ve 80/20 bolme
csv_path = Path(__file__).resolve().parent.parent / "data" / "ccpp.csv"
df = pd.read_csv(csv_path)

split = int(0.8 * len(df))
train, test = df.iloc[:split], df.iloc[split:]

features = ["AT", "V", "AP", "RH"]
X_train, y_train = train[features], train["PE"]
X_test, y_test = test[features], test["PE"]

# 1. Baseline (egitim ortalamasi)
baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))

# 2. Week 09 referans: tek degiskenli AT dogrusu
a_at, b_at = np.polyfit(train["AT"], y_train, 1)
at_mae = mean_absolute_error(y_test, a_at * test["AT"] + b_at)

# 3. Week 10 yontem (#1): Coklu Dogrusal Regresyon (OLS)
model = LinearRegression()
model.fit(X_train, y_train)

train_mae = mean_absolute_error(y_train, model.predict(X_train))
test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print(f"Intercept w0: {model.intercept_:.4f}")
for f, w in zip(features, model.coef_):
    print(f"Weight {f}: {w:.4f}")
print(f"\nBaseline Test MAE:          {baseline_mae:.2f} MW")
print(f"AT-only Line Test MAE:      {at_mae:.2f} MW")
print(f"Linear Regression Train MAE:{train_mae:.2f} MW")
print(f"Linear Regression Test MAE: {test_mae:.2f} MW (RMSE={test_rmse:.2f} MW, R2={test_r2:.3f})")
