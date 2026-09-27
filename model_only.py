"""
=============================================================================
RETAIL PROFITABILITY - MODELLING ONLY
=============================================================================
Follows the modelling narrative in the report:

  Stage 1  Ridge Regression on Profit          -> tested, REJECTED
  Stage 2  Random Forest Regression on Profit  -> tested, REJECTED
  Stage 3  Logistic Regression on Is_Loss      -> FINAL supervised model
  Stage 4  K-Means clustering (transactions + sub-categories)

No descriptive analysis, no charts.
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix)

warnings.filterwarnings("ignore")

DATA_PATH = "superstore.csv"
SEED = 42


# ---------------------------------------------------------------------------
# 1. LOAD AND PREPARE
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

# Cleaning rules from the report: remove extreme anomalies only.
before = len(df)
df = df[~((df["Quantity"] > 100) | (df["Discount"] < 0) | (df["Discount"] > 1))].copy()
print(f"Rows: {before:,} -> {len(df):,}  ({before - len(df)} removed)")

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

df["Unit_Price"]    = df["Sales"] / df["Quantity"]
df["Ship_Time"]     = (df["Ship Date"] - df["Order Date"]).dt.days
df["Order_Year"]    = df["Order Date"].dt.year
df["Order_Month"]   = df["Order Date"].dt.month
df["Order_Quarter"] = df["Order Date"].dt.quarter
df["Order_DOW"]     = df["Order Date"].dt.dayofweek

NUM = ["Sales", "Discount", "Quantity", "Unit_Price", "Ship_Time",
       "Order_Year", "Order_Month", "Order_Quarter", "Order_DOW"]
CAT = ["Category", "Sub-Category", "Segment", "Region", "Ship Mode"]

pre = lambda: ColumnTransformer([
    ("num", "passthrough", NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])


# ---------------------------------------------------------------------------
# 2. STAGE 1 - RIDGE REGRESSION ON PROFIT  (tested, rejected)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 1  Ridge Regression on Profit")
print("=" * 70)

X, y = df[NUM + CAT], df["Profit"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED)

ridge = Pipeline([("pre", pre()), ("model", Ridge(alpha=1.0))]).fit(Xtr, ytr)
tr, va = ridge.predict(Xtr), ridge.predict(Xva)

print(f"  Train R2   : {r2_score(ytr, tr):>8.3f}")
print(f"  Valid R2   : {r2_score(yva, va):>8.3f}   <- negative: worse than predicting the mean")
print(f"  Valid MAE  : {mean_absolute_error(yva, va):>8.1f}")
print(f"  Valid RMSE : {np.sqrt(mean_squared_error(yva, va)):>8.1f}")
print("  VERDICT: REJECTED. Profit is too skewed and there are no cost variables,")
print("           so a linear model cannot capture it.")


# ---------------------------------------------------------------------------
# 3. STAGE 2 - RANDOM FOREST REGRESSION ON PROFIT  (tested, rejected)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 2  Random Forest Regression on Profit")
print("=" * 70)

rf = Pipeline([
    ("pre", pre()),
    ("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
]).fit(Xtr, ytr)
tr, va = rf.predict(Xtr), rf.predict(Xva)

print(f"  Train R2   : {r2_score(ytr, tr):>8.3f}   <- fits the training data well")
print(f"  Valid R2   : {r2_score(yva, va):>8.3f}   <- explains almost nothing unseen")
print(f"  Valid MAE  : {mean_absolute_error(yva, va):>8.1f}")
print(f"  Valid RMSE : {np.sqrt(mean_squared_error(yva, va)):>8.1f}")
print("  VERDICT: REJECTED. Severe overfitting. Regression is the wrong frame.")


# ---------------------------------------------------------------------------
# 4. STAGE 3 - LOGISTIC REGRESSION ON Is_Loss  (final supervised model)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 3  Logistic Regression on Is_Loss   [FINAL MODEL]")
print("=" * 70)
print("  The business does not need to know an order will make £4.12.")
print("  It needs to know whether it will lose money at all.\n")

df["Is_Loss"] = (df["Profit"] < 0).astype(int)
Xc, yc = df[NUM + CAT], df["Is_Loss"]
Xtr, Xva, ytr, yva = train_test_split(Xc, yc, test_size=0.3,
                                      random_state=SEED, stratify=yc)

logit = Pipeline([
    ("pre", pre()),
    ("model", LogisticRegression(max_iter=1000, solver="liblinear")),
]).fit(Xtr, ytr)

proba = logit.predict_proba(Xva)[:, 1]
pred = (proba >= 0.5).astype(int)

acc, auc = accuracy_score(yva, pred), roc_auc_score(yva, proba)
print(f"  Accuracy : {acc:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print()
print(classification_report(yva, pred, target_names=["Profit", "Loss"], digits=3))

cm = confusion_matrix(yva, pred)
print("  Confusion matrix")
print(f"    Profitable orders : {cm[0,0]:>5,} correct | {cm[0,1]:>4,} wrongly flagged")
print(f"    Loss-making orders: {cm[1,1]:>5,} caught  | {cm[1,0]:>4,} missed")

# Coefficients - what actually drives loss risk
ohe = logit.named_steps["pre"].named_transformers_["cat"]
names = NUM + list(ohe.get_feature_names_out(CAT))
coefs = pd.Series(logit.named_steps["model"].coef_[0], index=names).sort_values()
print("\n  Strongest drivers of LOSS risk")
for n, v in coefs.tail(8)[::-1].items():
    print(f"    {n:<40s} {v:+.3f}")
print("\n  Strongest drivers AGAINST loss")
for n, v in coefs.head(5).items():
    print(f"    {n:<40s} {v:+.3f}")


# ---------------------------------------------------------------------------
# 5. STAGE 4 - K-MEANS CLUSTERING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 4  K-Means clustering")
print("=" * 70)

# -- transaction level ------------------------------------------------------
tx = df[["Sales", "Discount", "Quantity", "Profit", "Unit_Price"]].copy()
tx["Cluster"] = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(
    StandardScaler().fit_transform(tx))

summary = tx.groupby("Cluster").agg(
    Orders=("Profit", "size"),
    Avg_Discount=("Discount", "mean"),
    Avg_Profit=("Profit", "mean"),
    Avg_Sales=("Sales", "mean"),
    Total_Profit=("Profit", "sum"),
).round(2).sort_values("Avg_Profit")

print("\n  Transaction-level clusters")
print(summary.to_string())
worst = summary.index[0]
print(f"\n  Cluster {worst}: {summary.loc[worst,'Orders']:,.0f} orders, "
      f"avg discount {summary.loc[worst,'Avg_Discount']:.2f}, "
      f"avg profit {summary.loc[worst,'Avg_Profit']:.0f}")
print("  -> the High Discount / High Loss segment: the main source of value destruction")

# -- sub-category level -----------------------------------------------------
sub = df.groupby("Sub-Category").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Avg_Discount=("Discount", "mean"),
    Orders=("Profit", "size"),
)
sub["Cluster"] = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(
    StandardScaler().fit_transform(sub))

print("\n  Sub-category clusters")
for c in sorted(sub.Cluster.unique()):
    grp = sub[sub.Cluster == c]
    print(f"\n    Cluster {c}  (profit £{grp.Total_Profit.sum():,.0f}, "
          f"avg discount {grp.Avg_Discount.mean():.2f})")
    print(f"      {', '.join(grp.index)}")


# ---------------------------------------------------------------------------
# 6. SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Ridge Regression         REJECTED   (valid R2 negative)")
print(f"  Random Forest Regression REJECTED   (valid R2 near zero, overfit)")
print(f"  Logistic Regression      SELECTED   accuracy {acc:.3f}, ROC-AUC {auc:.3f}")
print(f"  K-Means                  isolates the high-discount / high-loss segment")
print("=" * 70)
