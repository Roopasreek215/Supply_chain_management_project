# ---------------------------
# Supply Chain Management Project
# ---------------------------

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px

# 2. Dataset Creation
np.random.seed(42)
num_rows = 1000
start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
dates = [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 730, num_rows)]
products = ['Product A', 'Product B', 'Product C', 'Product D']
suppliers = ['Supplier X', 'Supplier Y', 'Supplier Z']
warehouses = ['Warehouse 1', 'Warehouse 2']
categories = ['Electronics', 'Furniture', 'Clothing']

df = pd.DataFrame({
    "Date": dates,
    "Product": np.random.choice(products, num_rows),
    "Supplier": np.random.choice(suppliers, num_rows),
    "Warehouse": np.random.choice(warehouses, num_rows),
    "Category": np.random.choice(categories, num_rows),
    "Order_Demand": np.random.poisson(lam=20, size=num_rows),
    "Shipping_Delay_Days": np.random.randint(1, 15, num_rows),
    "Inventory_Level": np.random.randint(50, 500, num_rows)
})

# Sort and reset index
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# 3. Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# 4. Exploratory Data Analysis (EDA)
print("Basic Info:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Description:\n", df.describe())

# Demand Trend Line Chart
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Date', y='Order_Demand')
plt.title("Order Demand Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("demand_trend_plot.png")
plt.show()

# 5. Model Building (Random Forest Regressor)
features = df[['Month', 'Year']]
target = df['Order_Demand']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# 6. Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# 7. Save Predictions
results_df = X_test.copy()
results_df['Actual_Demand'] = y_test.values
results_df['Predicted_Demand'] = y_pred
results_df.to_csv("demand_predictions.csv", index=False)

# 8. Optional Interactive Plot (Plotly)
fig = px.line(df.sort_values("Date"), x='Date', y='Order_Demand', title='Order Demand Trend')
fig.show()

# Save dataset
df.to_csv("supply_chain_sample_data.csv", index=False)
