"""
Reproduces the Dataveil Phase 2 synthetic data and analyses.
Follows Appendix C (data generation), Appendix A (fraud), Appendix B (segmentation).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# PART 1 - TRANSACTION DATASET (N = 50,000; fraud_rate = 0.012)
# ----------------------------------------------------------------------
n_fraud, n_legit = 600, 49400

# Legitimate
legit = pd.DataFrame({
    "amount":           rng.lognormal(mean=4.4, sigma=0.9, size=n_legit),
    "hour":             rng.integers(6, 23, size=n_legit),
    "txn_velocity_1h":  rng.poisson(1.4, size=n_legit),
    "acct_age_days":    rng.integers(120, 4001, size=n_legit),
    "amount_to_avg_ratio": rng.normal(1.0, 0.35, size=n_legit),
    "cross_border":     rng.binomial(1, 0.07, size=n_legit),
    "is_fraud":         0,
})
# Fraud
fraud_hours = rng.choice([0,1,2,3,4,5,22,23], size=n_fraud)
fraud = pd.DataFrame({
    "amount":           rng.lognormal(mean=5.8, sigma=1.1, size=n_fraud),
    "hour":             fraud_hours,
    "txn_velocity_1h":  rng.poisson(5.5, size=n_fraud),
    "acct_age_days":    rng.integers(1, 401, size=n_fraud),
    "amount_to_avg_ratio": rng.normal(4.2, 1.6, size=n_fraud),
    "cross_border":     rng.binomial(1, 0.55, size=n_fraud),
    "is_fraud":         1,
})

txn = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
txn["amount_to_avg_ratio"] = txn["amount_to_avg_ratio"].clip(lower=0.05)
txn["amount"] = txn["amount"].round(2)
txn["amount_to_avg_ratio"] = txn["amount_to_avg_ratio"].round(3)

txn.to_csv("dataveil_transactions.csv", index=False)
print(f"Transactions: {len(txn):,} rows, fraud rate = {txn.is_fraud.mean()*100:.2f}%")

# ----------------------------------------------------------------------
# PART 2 - CLIENT DATASET (M = 12,000)
# ----------------------------------------------------------------------
M = 12000
age = np.clip(rng.normal(56, 14, size=M), 22, 92)
aum = np.clip(rng.lognormal(12.3, 0.85, size=M) * (1 + (age - 56)/120), 25000, 40_000_000)
risk0 = rng.integers(1, 11, size=M).astype(float)
risk = np.clip(risk0 - (age - 56)/12, 1, 10)
tenure = rng.integers(0, 25, size=M)
logins = rng.poisson(np.maximum(0.2, 3 + (10 - risk)*0.2 + tenure*0.1))
products = rng.integers(1, 8, size=M)

clients = pd.DataFrame({
    "age": age.round().astype(int),
    "aum": aum.round(2),
    "risk_tolerance": risk.round(1),
    "tenure_yrs": tenure,
    "logins_per_mo": logins,
    "products_held": products,
})
clients.to_csv("dataveil_clients.csv", index=False)
print(f"Clients: {len(clients):,} rows")

# ----------------------------------------------------------------------
# ANALYSIS A - FRAUD DETECTION
# ----------------------------------------------------------------------
feat = ["amount","hour","txn_velocity_1h","acct_age_days","amount_to_avg_ratio","cross_border"]
X = txn[feat].values
y = txn["is_fraud"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

# Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=0.012, random_state=SEED)
iso.fit(Xtr_s)
iso_pred = (iso.predict(Xte_s) == -1).astype(int)
iso_score = -iso.score_samples(Xte_s)

# Logistic Regression
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
lr.fit(Xtr_s, ytr)
lr_prob = lr.predict_proba(Xte_s)[:,1]
lr_pred = (lr_prob >= 0.5).astype(int)

print("\n--- FRAUD DETECTION (test n = {:,}) ---".format(len(yte)))
for name, pred, score in [("Isolation Forest", iso_pred, iso_score),
                          ("Logistic Regression", lr_pred, lr_prob)]:
    p = precision_score(yte, pred); r = recall_score(yte, pred)
    f = f1_score(yte, pred); auc = roc_auc_score(yte, score)
    print(f"{name:22s} P={p:.3f} R={r:.3f} F1={f:.3f} AUC={auc:.3f}")

cm = confusion_matrix(yte, iso_pred)
print("Isolation Forest confusion matrix [TN FP / FN TP]:")
print(cm)
print(f"Actual fraud in test set: {yte.sum()}")

# ----------------------------------------------------------------------
# ANALYSIS B - SEGMENTATION
# ----------------------------------------------------------------------
cfeat = ["age","aum","risk_tolerance","tenure_yrs","logins_per_mo","products_held"]
Xc = StandardScaler().fit_transform(clients[cfeat].values)

print("\n--- CLUSTER SELECTION ---")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xc)
    sil = silhouette_score(Xc, km.labels_)
    print(f"k={k}  inertia={km.inertia_:10.0f}  silhouette={sil:.3f}")

km4 = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Xc)
clients["segment"] = km4.labels_
prof = clients.groupby("segment").agg(
    n=("age","size"), mean_age=("age","mean"), mean_aum=("aum","mean"),
    mean_risk=("risk_tolerance","mean"), mean_tenure=("tenure_yrs","mean"),
    mean_logins=("logins_per_mo","mean"), mean_products=("products_held","mean"))
print("\n--- SEGMENT PROFILES (k=4) ---")
print(prof.round(1).to_string())

pca = PCA(n_components=2).fit(Xc)
print(f"\nPCA variance explained by 2 PCs: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")
