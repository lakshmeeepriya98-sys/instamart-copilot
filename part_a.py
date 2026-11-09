import os
import sqlite3
import numpy as np
import pandas as pd

os.makedirs("data/clean", exist_ok=True)
os.makedirs("sql", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

# 1. Clean Customers
cust = pd.read_csv("data/raw/customers.csv")
cust["city"] = cust["city"].astype(str).str.strip().str.title()
cust["gender"] = cust["gender"].apply(lambda g: "male" if str(g).strip().lower() in ["male", "m"] else ("female" if str(g).strip().lower() in ["female", "f"] else "unknown"))
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median")).astype(int)
cust["signup_date"] = pd.to_datetime(cust["signup_date"])

# 2. Clean Orders
orders = pd.read_csv("data/raw/orders.csv", keep_default_na=False)
prod = pd.read_csv("data/raw/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders = orders.drop_duplicates()
orders = orders.sort_values(by="order_ts").drop_duplicates(subset=["order_id"], keep="last")

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce").fillna(0)
orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")

merged = orders.merge(prod[["product_id", "price_inr", "category"]], on="product_id", how="left")
na_amt = merged["order_amount"].isna()
orders.loc[na_amt, "order_amount"] = np.round(
    merged.loc[na_amt, "price_inr"] * merged.loc[na_amt, "quantity"] * (1 - merged.loc[na_amt, "discount_pct"] / 100.0), 2
)

orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

orders["category"] = orders["product_id"].map(prod.set_index("product_id")["category"].to_dict())
cat_meds = orders[orders["status"] == "completed"].groupby("category")["delivery_minutes"].median()
for c, med in cat_meds.items():
    m = (orders["status"] == "completed") & (orders["category"] == c) & (orders["delivery_minutes"].isna())
    orders.loc[m, "delivery_minutes"] = med
orders = orders.drop(columns=["category"])
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# 3. Save Clean CSVs & DB
cust.to_csv("data/clean/customers.csv", index=False)
prod.to_csv("data/clean/products.csv", index=False)
orders.to_csv("data/clean/orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
cust.to_sql("customers", conn, if_exists="replace", index=False)
prod.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

# 4. Checkpoints
assert len(cust) == 1200
assert cust["age"].isna().sum() == 0
assert len(orders) == 12309
assert (orders["status"] == "pending").sum() == 0
assert (orders["status"] == "completed").sum() == 11110
assert (orders["status"] == "cancelled").sum() == 1199
assert orders["order_amount"].isna().sum() == 0
assert round(orders[orders["status"] == "completed"]["order_amount"].sum(), 2) == 2998668.80

print("ALL CHECKPOINTS PASSED")
