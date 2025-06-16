# Supply_chain_management_project
**Summary of the project**:

1. Generated a synthetic dataset with 1,000 entries simulating supply chain operations.
2. Columns include Product, Supplier, Warehouse, Category, Order Demand, Shipping Delay, and Inventory Level.
3. Converted the Date column to datetime and extracted Month and Year for trend analysis.
4. Conducted EDA to examine demand, inventory levels, and shipping delays over time.
5. Visualized demand trends using a line plot (Matplotlib & Plotly).
6. Used `RandomForestRegressor` to predict future `Order_Demand`.
7. Split data into training and testing sets (80/20 split).
8. Evaluated model performance using RMSE (Root Mean Squared Error).
9. Exported predicted vs actual values to a CSV file.
10. Final files include dataset, demand trend plot, and prediction results.
