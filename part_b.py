import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

orders = pd.read_csv("data/clean/orders.csv")
cust = pd.read_csv("data/clean/customers.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# --- Task 1: RFM ---
completed = orders[orders["status"] == "completed"].copy()
snap = datetime(2025, 10, 1)
rfm = completed.groupby("customer_id").agg(
    recency=("order_ts", lambda s: (snap - s.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# --- Task 2 & 3: Clustering ---
scaler = StandardScaler()
X_rfm = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

km = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = km.fit_predict(X_rfm)

seg_means = rfm.groupby("segment_id")["monetary"].mean()
prem_id = seg_means.idxmax()
rfm["is_premium"] = rfm["segment_id"] == prem_id

names = {prem_id: "High-Value Champions"}
rem = [i for i in range(4) if i != prem_id]
names[rem[0]] = "At-Risk Occasionals"
names[rem[1]] = "Frequent Everyday"
names[rem[2]] = "Occasional Low-Spend"
rfm["segment_name"] = rfm["segment_id"].map(names)

rfm.to_csv("outputs/customer_segments.csv", index=False)

# --- Task 4-6: Lapse Prediction ---
cutoff = datetime(2025, 9, 1)
pre_orders = orders[orders["order_ts"] < cutoff].copy()
pre_comp = pre_orders[pre_orders["status"] == "completed"].copy()

cohort_ids = pre_comp["customer_id"].unique()
cohort = pd.DataFrame({"customer_id": cohort_ids})

post_comp = orders[(orders["order_ts"] >= cutoff) & (orders["status"] == "completed")]
active_post = set(post_comp["customer_id"].unique())
cohort["lapsed"] = cohort["customer_id"].apply(lambda cid: 0 if cid in active_post else 1)

rfm_pre = pre_comp.groupby("customer_id").agg(
    recency=("order_ts", lambda s: (cutoff - s.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

tot_orders = pre_orders.groupby("customer_id")["order_id"].count()
canc_orders = pre_orders[pre_orders["status"] == "cancelled"].groupby("customer_id")["order_id"].count()
canc_rate = (canc_orders / tot_orders).fillna(0).reset_index(name="cancellation_rate")

prod = pd.read_csv("data/clean/products.csv")
pre_comp_prod = pre_comp.merge(prod, on="product_id")
fav_cat = pre_comp_prod.groupby("customer_id")["category"].agg(lambda s: s.mode()[0]).reset_index(name="fav_category")

feats = cohort.merge(rfm_pre, on="customer_id").merge(canc_rate, on="customer_id").merge(fav_cat, on="customer_id")
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
cust["tenure"] = (cutoff - cust["signup_date"]).dt.days
feats = feats.merge(cust[["customer_id", "city", "age", "gender", "tenure"]], on="customer_id")

num_cols = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery", "cancellation_rate", "age", "tenure"]
cat_cols = ["fav_category", "city", "gender"]

X = feats[num_cols + cat_cols]
y = feats["lapsed"]

X_train, X_test, y_train, y_test, id_tr, id_te = train_test_split(
    X, y, feats["customer_id"], test_size=0.2, stratify=y, random_state=42
)

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([
    ("prep", preproc),
    ("clf", RandomForestClassifier(random_state=42))
])

param_grid = {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test F1: {f1_score(y_test, y_pred):.4f}")

joblib.dump(best_model, "outputs/lapse_model.joblib")

test_res = pd.DataFrame({
    "customer_id": id_te,
    "lapse_probability": best_model.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.values
})
prem_cids = set(rfm[rfm["is_premium"]]["customer_id"])
prem_test = test_res[test_res["customer_id"].isin(prem_cids)].sort_values(by="lapse_probability", ascending=False)
prem_test.head(10).to_csv("outputs/retention_list.csv", index=False)
print("Part B Done.")
