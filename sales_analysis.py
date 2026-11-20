from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path("data/sales_data.csv")
OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)

# 1. Load and inspect
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe())

# 2. Clean
df = df.drop_duplicates().copy()
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Customer_Name"] = df["Customer_Name"].fillna("Unknown")
df["Region"] = df["Region"].fillna("Unknown")
for col in ["Quantity", "Unit_Price", "Discount", "Cost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Discount"] = df["Discount"].fillna(0)
df = df.dropna(subset=["Order_Date", "Quantity", "Unit_Price", "Cost"])
df["Revenue"] = (df["Quantity"] * df["Unit_Price"] * (1-df["Discount"])).round(2)
df["Profit"] = (df["Revenue"] - df["Cost"]).round(2)
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# 3. KPIs
revenue = df["Revenue"].sum()
profit = df["Profit"].sum()
orders = df["Order_ID"].nunique()
aov = revenue / orders
margin = profit / revenue * 100
print("\n===== BUSINESS KPIs =====")
print(f"Total revenue: ₹{revenue:,.2f}")
print(f"Total profit: ₹{profit:,.2f}")
print(f"Unique orders: {orders:,}")
print(f"Units sold: {df['Quantity'].sum():,}")
print(f"Average order value: ₹{aov:,.2f}")
print(f"Profit margin: {margin:.2f}%")

# 4. Grouped analysis
category = df.groupby("Category").agg(Revenue=("Revenue","sum"), Profit=("Profit","sum"), Units_Sold=("Quantity","sum"), Orders=("Order_ID","nunique")).sort_values("Revenue",ascending=False)
region = df.groupby("Region").agg(Revenue=("Revenue","sum"),Profit=("Profit","sum")).sort_values("Revenue",ascending=False)
monthly = df.groupby("Month",as_index=False).agg(Revenue=("Revenue","sum"),Profit=("Profit","sum"))
products = df.groupby("Product").agg(Revenue=("Revenue","sum"),Profit=("Profit","sum")).sort_values("Revenue",ascending=False)
customers = df.groupby("Customer_Name").agg(Revenue=("Revenue","sum"),Orders=("Order_ID","nunique")).sort_values("Revenue",ascending=False)
print("\nCategory performance:\n",category.round(2))
print("\nRegion performance:\n",region.round(2))
print("\nMonthly sales:\n",monthly.round(2))
print("\nTop products:\n",products.head().round(2))
print("\nTop customers:\n",customers.head(10).round(2))

# 5. Visualizations (Matplotlib)
plt.figure(figsize=(11,5))
plt.plot(monthly["Month"],monthly["Revenue"],marker="o")
plt.title("Monthly Revenue Trend - 2024"); plt.xlabel("Month"); plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(OUT/"monthly_revenue_trend.png",dpi=150); plt.close()

plt.figure(figsize=(9,5))
cat_plot=category.sort_values("Revenue")
plt.barh(cat_plot.index,cat_plot["Revenue"])
plt.title("Revenue by Product Category"); plt.xlabel("Revenue (INR)"); plt.tight_layout()
plt.savefig(OUT/"revenue_by_category.png",dpi=150); plt.close()

plt.figure(figsize=(8,5))
reg_plot=region.sort_values("Profit")
plt.bar(reg_plot.index,reg_plot["Profit"])
plt.title("Profit by Region"); plt.xlabel("Region"); plt.ylabel("Profit (INR)")
plt.tight_layout(); plt.savefig(OUT/"profit_by_region.png",dpi=150); plt.close()

plt.figure(figsize=(9,5))
x=np.arange(len(category.index)); width=.36
plt.bar(x-width/2,category["Revenue"],width,label="Revenue")
plt.bar(x+width/2,category["Profit"],width,label="Profit")
plt.xticks(x,category.index,rotation=20); plt.title("Revenue and Profit by Category")
plt.ylabel("Amount (INR)"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"revenue_profit_by_category.png",dpi=150); plt.close()

# 6. Export analysis outputs
df.to_csv("data/cleaned_sales_data.csv",index=False)
category.to_csv("data/category_summary.csv")
region.to_csv("data/region_summary.csv")
monthly.to_csv("data/monthly_sales_summary.csv",index=False)
print("\nAnalysis completed. Check data/ and visualizations/.")
