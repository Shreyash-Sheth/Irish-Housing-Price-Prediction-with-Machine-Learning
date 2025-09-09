#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.compose import TransformedTargetRegressor

import joblib


# In[3]:


CSV_PATH = "C:\\Users\\yashs\\Desktop\\Project\\Machine Learning\\irish_property_listings.csv"

df_raw = pd.read_csv(CSV_PATH)
print(df_raw.shape)
df_raw.head(3)


# In[4]:


df = df_raw.copy()
df.info()


# In[5]:


# % of missing values per column
(df.isna().mean().sort_values(ascending=False) * 100).round(1)


# In[6]:


df = df_raw.copy()

# Price
def parse_price(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    # Remove currency symbols and non-digits
    s = re.sub(r"[^\d.]", "", s)
    try:
        return float(s) if s != "" else np.nan
    except:
        return np.nan

df["price_num"] = df["price"].apply(parse_price)


# In[7]:


# AMV flag
if "AMVprice" in df.columns:
    df["is_amv"] = df["AMVprice"].notna() & (df["AMVprice"].astype(str).str.strip() != "")
else:
    df["is_amv"] = df["price"].astype(str).str.contains("AMV", case=False, na=False)

df["is_amv"] = df["is_amv"].astype(int) 


# In[8]:


# Dates & listing age
if "publishDate" in df.columns:
    df["publishDate"] = pd.to_datetime(df["publishDate"], errors="coerce")
    # Use the max publishDate in the dataset as "today" reference to avoid time leakage
    ref_date = df["publishDate"].max()
    df["listing_age_days"] = (ref_date - df["publishDate"]).dt.days
else:
    df["listing_age_days"] = np.nan


# In[9]:


# Property size numeric
def parse_size(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    # pull the first number (could be float)
    m = re.search(r"(\d+(\.\d+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else np.nan

df["propertySize_num"] = df["propertySize"].apply(parse_size)


# In[10]:


# Booleans to 0/1
for col in ["m_hasVideo", "m_hasVirtualTour", "m_hasBrochure"]:
    if col in df.columns:
        df[col] = df[col].map({True:1, False:0, "True":1, "False":0, 1:1, 0:0})
        df[col] = df[col].fillna(0).astype(int)
    else:
        df[col] = 0  # if missing entirely, set to 0


# In[11]:


# --- Images ---
if "m_totalImages" not in df.columns:
    df["m_totalImages"] = 0
df["m_totalImages"] = pd.to_numeric(df["m_totalImages"], errors="coerce")


# In[12]:


# --- BER mapping (ordered A1 (best) -> G (worst)) ---
ber_map = {
    "A1":15,"A2":14,"A3":13,
    "B1":12,"B2":11,"B3":10,
    "C1":9,"C2":8,"C3":7,
    "D1":6,"D2":5,
    "E1":4,"E2":3,
    "F":2,"G":1
}


# In[13]:


# Clean typical oddities:
def clean_ber(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper()
    s = s.replace("A1A2", "A2")  # fallback
    if s in ["XXX", "SI666", "UNKNOWN", "N/A", "NA", ""]:
        return np.nan
    return s

df["ber_clean"] = df["ber_rating"].apply(clean_ber)
df["ber_score"] = df["ber_clean"].map(ber_map)  # numeric, higher = better


# In[14]:


# --- featuredLevel/category default ---
df["featuredLevel"] = df.get("featuredLevel", "standard").fillna("standard").str.lower()
df["category"] = df.get("category", "Buy").fillna("Buy")


# In[15]:


# --- Bedrooms/Bathrooms numeric ---
for col in ["numBedrooms","numBathrooms"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan


# In[16]:


# --- Lat/Long numeric ---
for col in ["latitude","longitude"]:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")


# In[17]:


df[["price","price_num","is_amv","publishDate","listing_age_days","propertySize","propertySize_num","ber_rating","ber_clean","ber_score"]].head(10)


# In[18]:


print("Rows:", len(df))
print("Time range:", df["publishDate"].min(), "to", df["publishDate"].max())
print("Price (clean) describe:")
display(df["price_num"].describe())

# Missingness snapshot
missing_pct = (df.isna().mean()*100).sort_values(ascending=False).round(1)
missing_pct.head(15)


# In[19]:


# Price distribution
plt.figure(figsize=(6,4))
sns.histplot(df["price_num"].dropna(), bins=60)
plt.title("Price distribution (raw)")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(np.log1p(df["price_num"].dropna()), bins=60)
plt.title("Price distribution (log1p)")
plt.show()


# In[20]:


# Numeric correlations
num_cols = ["price_num","numBedrooms","numBathrooms","propertySize_num","m_totalImages","ber_score","listing_age_days","latitude","longitude"]
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap (numeric)")
plt.show()


# In[21]:


# Haversine distance in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Major city centres
cities = {
    "Dublin":   (53.3498, -6.2603),
    "Cork":     (51.8985, -8.4756),
    "Galway":   (53.2707, -9.0568),
    "Limerick": (52.6638, -8.6267),
    "Waterford":(52.2593, -7.1101)
}

for city, (clat, clon) in cities.items():
    df[f"dist_to_{city.lower()}"] = haversine(df["latitude"], df["longitude"], clat, clon)

dist_cols = [c for c in df.columns if c.startswith("dist_to_")]
df[["latitude","longitude"] + dist_cols].head(3)


# In[22]:


irish_counties = [
    "Dublin","Cork","Galway","Limerick","Waterford","Wexford","Wicklow","Kildare","Meath","Louth",
    "Kilkenny","Kerry","Clare","Tipperary","Offaly","Laois","Westmeath","Longford","Leitrim",
    "Roscommon","Mayo","Sligo","Donegal","Monaghan","Cavan","Carlow"
]

def find_county_from_title(s):
    if pd.isna(s): return np.nan
    s = str(s)
    for county in irish_counties:
       
        if re.search(rf"\b{county}\b", s, flags=re.IGNORECASE):
            return county
    return np.nan

df["county_guess"] = df["title"].apply(find_county_from_title)
df["county_guess"].value_counts(dropna=False).head(10)


# In[23]:


# Keep only rows with a valid price
data = df[df["price_num"].notna()].copy()


lower = data["price_num"].quantile(0.005)
upper = data["price_num"].quantile(0.995)
data["price_num"] = data["price_num"].clip(lower, upper)


bins = pd.qcut(data["price_num"], q=10, duplicates="drop")
X = data.drop(columns=["price_num"])
y = data["price_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=bins
)

X_train.shape, X_test.shape


# In[24]:


num_features = [
    "numBedrooms","numBathrooms","propertySize_num","m_totalImages","ber_score","listing_age_days",
    "latitude","longitude",
    "dist_to_dublin","dist_to_cork","dist_to_galway","dist_to_limerick","dist_to_waterford"
]
bin_features = ["m_hasVideo","m_hasVirtualTour","m_hasBrochure","is_amv"]
cat_features = ["propertyType","category","sellerType","featuredLevel","county_guess"]

# Make sure missing columns exist
for col in num_features + bin_features + cat_features:
    if col not in X_train.columns:
        X_train[col] = np.nan
        X_test[col] = np.nan

# Preprocessors
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())  # helps linear models; trees will cope fine even if scaled
])

binary_transformer = "passthrough"  # already 0/1

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("bin", binary_transformer, bin_features),
        ("cat", categorical_transformer, cat_features),
    ],
    remainder="drop"
)


# In[25]:


def evaluate(model, X_tr, y_tr, X_te, y_te, name="model"):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    rmse = mean_squared_error(y_te, preds, squared=False)
    r2 = r2_score(y_te, preds)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []


# In[26]:


# 1) Dummy baseline (median)
dummy = Pipeline([("preprocess", preprocess), ("model", DummyRegressor(strategy="median"))])
results.append(evaluate(dummy, X_train, y_train, X_test, y_test, "Baseline_median"))


# In[27]:


# 2) Linear Regression with log1p target
lin = Pipeline([("preprocess", preprocess), ("model", LinearRegression())])
lin_ttr = TransformedTargetRegressor(regressor=lin, func=np.log1p, inverse_func=np.expm1)
results.append(evaluate(lin_ttr, X_train, y_train, X_test, y_test, "LinearRegression_logTarget"))


# In[28]:


# 3) Random Forest
rf = Pipeline([("preprocess", preprocess),
               ("model", RandomForestRegressor(
                   n_estimators=300, max_depth=None, min_samples_split=4,
                   random_state=42, n_jobs=-1
               ))])
results.append(evaluate(rf, X_train, y_train, X_test, y_test, "RandomForest"))


# In[29]:


# 4) HistGradientBoosting
hgb = Pipeline([("preprocess", preprocess),
                ("model", HistGradientBoostingRegressor(
                    max_depth=None, learning_rate=0.06, max_iter=500,
                    l2_regularization=0.0, random_state=42
                ))])
# Use log target for HGB too (often helps)
hgb_ttr = TransformedTargetRegressor(regressor=hgb, func=np.log1p, inverse_func=np.expm1)
results.append(evaluate(hgb_ttr, X_train, y_train, X_test, y_test, "HGB_logTarget"))

pd.DataFrame(results).sort_values("MAE")


# In[30]:


from scipy.stats import randint

rf_base = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_distributions = {
    "model__n_estimators": randint(200, 600),
    "model__max_depth": randint(6, 30),
    "model__min_samples_split": randint(2, 10),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["auto", "sqrt", 0.3, 0.5, 0.7]
}

rf_search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)
print("Best params:", rf_search.best_params_)
best_rf = rf_search.best_estimator_

# Evaluate tuned RF
preds = best_rf.predict(X_test)
print("Tuned RF  | MAE:", mean_absolute_error(y_test, preds),
      "RMSE:", mean_squared_error(y_test, preds, squared=False),
      "R2:", r2_score(y_test, preds))


# In[40]:


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Refit RandomForest on training data
rf_model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=15)
rf_model.fit(
    best_rf.named_steps["preprocess"].transform(X_train),  # transformed features
    y_train
)

# Run permutation importance on the pipeline
perm = permutation_importance(
    best_rf, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

# Get feature names after preprocessing
raw_feature_names = best_rf.named_steps["preprocess"].get_feature_names_out()

# Align sizes
n_features = min(len(raw_feature_names), len(perm.importances_mean))
feature_names = raw_feature_names[:n_features]

# Build dataframe
importances = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean[:n_features],
    "importance_std": perm.importances_std[:n_features]
}).sort_values("importance_mean", ascending=False)

# Make feature names more readable
importances["feature"] = (
    importances["feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("bin__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# Show top 15 features
print(importances.head(15))


# In[41]:


plt.figure(figsize=(8,10))
sns.barplot(
    data=importances.head(20),
    y="feature",
    x="importance_mean",
    xerr=importances.head(20)["importance_std"]
)
plt.title("Top 20 Most Important Features (Random Forest)")
plt.xlabel("Decrease in R² when feature is permuted")
plt.ylabel("")
plt.tight_layout()
plt.show()


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")

# Boxplot: Price by property type
plt.figure(figsize=(10,6))
sns.boxplot(
    data=df, 
    x="propertyType", 
    y="price", 
    showfliers=False,   # hides extreme outliers for readability
    palette="Set2"
)
plt.title("Distribution of Property Prices by Property Type")
plt.ylabel("Price (€)")
plt.xlabel("Property Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[43]:


# Scatterplot: Property size vs Price
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df, 
    x="propertySize", 
    y="price", 
    hue="propertyType",   # color by property type
    alpha=0.6
)
plt.title("Property Size vs Price")
plt.ylabel("Price (€)")
plt.xlabel("Property Size (sq. meters)")
plt.legend(title="Property Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[ ]:




