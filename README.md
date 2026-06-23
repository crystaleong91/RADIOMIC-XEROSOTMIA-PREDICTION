# RADIOMIC-XEROSOTMIA-PREDICTION
Temporal Megavoltage Computed Tomography Radiomics for Prediction of Late Xerostomia Following Radiotherapy for Head and Neck Cancer
Longitudinal Analysis Across Weekly MVCT Scans
The same radiomics modelling pipeline was independently applied to weekly MVCT datasets acquired during radiotherapy (Weeks 2–7) 6 months, 12 months and 24
months. Each weekly dataset was analysed using an identical workflow consisting of feature reduction, model training, and validation procedures.
Feature reduction was performed using a fixed correlation threshold (|ρ| &gt; 0.85), which was consistently applied across all weekly datasets to ensure uniform feature preselection and reduce multicollinearity. 
Feature selection was performed using an L1-regularised logistic regression model.
Hyperparameter tuning was conducted independently within each weekly dataset using stratified k-fold cross-validation to identify the optimal regularisation strength.
The same modelling framework and cross-validation strategy were applied across all weekly datasets. To ensure stability of model estimation while accounting for
variations in event distribution, a predefined rule was used to select either 5-fold or 10-fold stratified cross-validation based on the number of events in each time point (6 months, 12 months, 24 months) of dataset. This approach ensured methodological consistency across time points while maintaining sufficient training sample sizes within each fold.
Importantly, all hyperparameter tuning and feature selection were performed exclusively within the training data for each weekly dataset. No information from the
test set was used during model optimisation, ensuring strict separation between training and evaluation phases.
Differences in selected features and optimal regularisation parameters across time points reflect inherent variability in radiomic feature distributions and event
incidence, rather than changes in the modelling strategy.
For model evaluation, performance was assessed using stratified cross-validation within the training cohort and validated on an independent test set.

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATA
# ============================================================
file_path = r"C:\Users\philipwyh86\OneDrive\Desktop\LWC personal\MVCT RADIOMIC\1. RETROSPECTIVE -191\MVCT_RETRO ANALYSIS\3. DELTA FEATURES\6M_XER\W1+D1.xlsx"
df = pd.read_excel(file_path)

TARGET = "6M_XEROSTOMIA"
df = df.dropna(subset=[TARGET])

df_num = df.select_dtypes(include=[np.number])
X_raw = df_num.drop(columns=[TARGET])
y = df_num[TARGET]

print("Loaded dataset:", df.shape)

# ============================================================
# 2. TRAIN–TEST SPLIT
# ============================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.30, stratify=y, random_state=42
)

# ============================================================
# 3. IMPUTATION + SCALING (TRAIN FIT ONLY)
# ============================================================
imp = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(imp.fit_transform(X_train_raw)),
    columns=X_raw.columns
)
X_test = pd.DataFrame(
    scaler.transform(imp.transform(X_test_raw)),
    columns=X_raw.columns
)

# ============================================================
# 4. CORRELATION FILTERING (TRAIN ONLY)
# ============================================================
CORR_THRESHOLD = 0.85

corr = X_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]

X_train_corr = X_train.drop(columns=to_drop)
X_test_corr  = X_test.drop(columns=to_drop, errors="ignore")

print("Features after correlation filtering:", X_train_corr.shape[1])

# ============================================================
# 5. LASSO FEATURE SELECTION (CV = 5)
# ============================================================
lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-1, 5, 100),
    penalty="l1",
    solver="saga",
    scoring="roc_auc",
    cv=5,
    max_iter=10000,
    random_state=42,
    n_jobs=-1
)
lasso_cv.fit(X_train_corr, y_train)

coef = lasso_cv.coef_.ravel()

importance = pd.DataFrame({
    "Feature": X_train_corr.columns,
    "Coefficient": coef
})

LASSO_THRESHOLD = 0.10

selected = importance.loc[
    importance["Coefficient"].abs() >= LASSO_THRESHOLD
].sort_values("Coefficient", key=np.abs, ascending=False)

selected_features = selected["Feature"].tolist()
print("Selected features:", len(selected_features))

X_train_lasso = X_train_corr[selected_features]
X_test_lasso  = X_test_corr[selected_features]

# ============================================================
# >>> PRINT OPTIMAL LAMBDA (λ)
# ============================================================
best_C = lasso_cv.C_[0]
best_lambda = 1 / best_C

print("\nOPTIMAL REGULARISATION PARAMETER")
print(f"Optimal C (inverse λ) = {best_C:.6f}")
print(f"Optimal λ (lambda)   = {best_lambda:.6f}")

# ============================================================
# 6. OVERSAMPLING (TRAIN ONLY)
# ============================================================
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_lasso, y_train)

# ============================================================
# 7. FINAL LASSO MODEL
# ============================================================
model = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=lasso_cv.C_[0],
    class_weight="balanced",
    max_iter=10000,
    random_state=42,
)
model.fit(X_train_bal, y_train_bal)

# ============================================================
# 8. CROSS-VALIDATED AUC (TRAIN)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X_train_lasso, y_train, cv=cv, scoring="roc_auc")

print(f"\nCV AUC = {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ============================================================
# 9. TRAIN PERFORMANCE + 95% CI
# ============================================================
train_probs = model.predict_proba(X_train_lasso)[:, 1]
train_auc = roc_auc_score(y_train, train_probs)

boot_train_auc = []
for i in range(10000):
    idx = resample(range(len(y_train)), replace=True)
    if len(np.unique(y_train.iloc[idx])) < 2:
        continue
    boot_train_auc.append(
        roc_auc_score(y_train.iloc[idx], train_probs[idx])
    )

train_ci_low, train_ci_high = np.percentile(boot_train_auc, [2.5, 97.5])

# ============================================================
# 10. YOUDEN THRESHOLD (TRAIN)
# ============================================================
fpr_train, tpr_train, thresholds = roc_curve(y_train, train_probs)
best_thresh = thresholds[np.argmax(tpr_train - fpr_train)]

train_pred = (train_probs >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()

train_sens = tp / (tp + fn)
train_spec = tn / (tn + fp)
train_acc  = accuracy_score(y_train, train_pred)

# ============================================================
# 11. TEST PERFORMANCE + 95% CI
# ============================================================
test_probs = model.predict_proba(X_test_lasso)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)

test_pred = (test_probs >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

test_sens = tp / (tp + fn)
test_spec = tn / (tn + fp)
test_acc  = accuracy_score(y_test, test_pred)

boot_test_auc = []
for i in range(10000):
    idx = resample(range(len(y_test)), replace=True)
    if len(np.unique(y_test.iloc[idx])) < 2:
        continue
    boot_test_auc.append(
        roc_auc_score(y_test.iloc[idx], test_probs[idx])
    )

test_ci_low, test_ci_high = np.percentile(boot_test_auc, [2.5, 97.5])

# ============================================================
# 12. PRINT RESULTS
# ============================================================
print("\nTRAIN PERFORMANCE")
print(f"AUC         = {train_auc:.3f}")
print(f"95% CI      = [{train_ci_low:.3f}, {train_ci_high:.3f}]")
print(f"Sensitivity = {train_sens:.3f}")
print(f"Specificity = {train_spec:.3f}")
print(f"Accuracy    = {train_acc:.3f}")
print(f"Threshold   = {best_thresh:.3f}")

print("\nTEST PERFORMANCE")
print(f"AUC         = {test_auc:.3f}")
print(f"95% CI      = [{test_ci_low:.3f}, {test_ci_high:.3f}]")
print(f"Sensitivity = {test_sens:.3f}")
print(f"Specificity = {test_spec:.3f}")
print(f"Accuracy    = {test_acc:.3f}")
print(f"Threshold   = {best_thresh:.3f}")

# ============================================================
# 13. ROC CURVE
# ============================================================
plt.figure(figsize=(6, 5))
plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC={train_auc:.3f})", lw=2)
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC={test_auc:.3f})", lw=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Week 2 Δ-Radiomics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 14. LASSO FEATURE SELECTION PLOT
# ============================================================
plt.figure(figsize=(10, max(4, 0.35 * len(selected))))
bars = plt.barh(
    selected["Feature"],
    selected["Coefficient"],
    color=["#1f77b4" if c > 0 else "#d62728" for c in selected["Coefficient"]]
)
plt.xlabel("LASSO Coefficient")
plt.title("LASSO-Selected Features (Week 2)")
plt.gca().invert_yaxis()
plt.grid(True, axis="x", alpha=0.3)

for bar, coef in zip(bars, selected["Coefficient"]):
    x = bar.get_width()
    plt.text(
        x + (0.02 if x > 0 else 0.00),
        bar.get_y() + bar.get_height()/2,
        f"{coef:.3f}",
        va="center",
        ha="left" if x > 0 else "left",
        fontsize=9
    )

plt.tight_layout()
plt.show()

# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

save_dir = r"C:\Users\philipwyh86\OneDrive\Desktop"
os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------
# 1. Selected features + coefficients
# ------------------------------------------------------------
selected.to_excel(
    os.path.join(save_dir, "SELECTED_FEATURES.xlsx"),
    index=False
)

# ------------------------------------------------------------
# 2. Training set (after LASSO + oversampling)
# ------------------------------------------------------------
pd.DataFrame(X_train_bal).assign(
    Target=y_train_bal.values
).to_excel(
    os.path.join(save_dir, "TRAIN_SELECTED.xlsx"),
    index=False
)

# ------------------------------------------------------------
# 3. Test set (selected features only, untouched)
# ------------------------------------------------------------
pd.DataFrame(X_test_lasso).assign(
    Target=y_test.values
).to_excel(
    os.path.join(save_dir, "TEST_SELECTED.xlsx"),
    index=False
)

print("\nSaved: Selected features, Train + Test sets")

# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

save_dir = r"C:\Users\philipwyh86\OneDrive\Desktop"
os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------
# 1. Selected features + coefficients
# ------------------------------------------------------------
selected.to_excel(
    os.path.join(save_dir, "SELECTED_FEATURES.xlsx"),
    index=False
)

# ------------------------------------------------------------
# 2. Training set (after LASSO + oversampling)
# ------------------------------------------------------------
pd.DataFrame(X_train_bal).assign(
    Target=y_train_bal.values
).to_excel(
    os.path.join(save_dir, "TRAIN_SELECTED.xlsx"),
    index=False
)

# ------------------------------------------------------------
# 3. Test set (selected features only, untouched)
# ------------------------------------------------------------
pd.DataFrame(X_test_lasso).assign(
    Target=y_test.values
).to_excel(
    os.path.join(save_dir, "TEST_SELECTED.xlsx"),
    index=False
)

print("\nSaved: Selected features, Train + Test sets")
