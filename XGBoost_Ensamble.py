import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn / Imblearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

# Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# For reproducibility
RANDOM_SEED = 42

##############################################################################
# 1) LOAD AND PREPARE DATA
##############################################################################
df = pd.read_csv('gr_features_by_object.csv', index_col=False)
constructed_df = pd.read_csv('new_features.csv', index_col=False)
merged = constructed_df.merge(df, on='name', how='inner')

valid_types = ['Q', 'QR', 'QX']
qso_select_df = merged[merged['type'].isin(valid_types)].copy()
qso_select_df.dropna(inplace=True)

# ------------------------------------------------------------------------------
# 1.a) Example: Additional / advanced features
#     In real usage, you might compute Lomb-Scargle periods, wavelet energies, etc.
#     We'll just demonstrate a few placeholders here:
# ------------------------------------------------------------------------------
qso_select_df['variance_ratio'] = (
    qso_select_df['Meanvariance_g'] / (qso_select_df['Meanvariance_r'] + 1e-8)
)
qso_select_df['mean_band_ratio'] = (
    qso_select_df['Mean_g'] / (qso_select_df['Mean_r'] + 1e-8)
)
qso_select_df['magnitude_diff'] = qso_select_df['Mean_g'] - qso_select_df['Mean_r']

# Placeholder advanced time-series feature (hypothetical):
# qso_select_df['ls_period_g'] = ...   # Lomb-Scargle period in g-band
# qso_select_df['ls_period_r'] = ...   # Lomb-Scargle period in r-band
# qso_select_df['wavelet_energy_g'] = ...
# etc.

X = qso_select_df.drop(columns=['name', 'type'])
y_str = qso_select_df['type']

# Train/test split
X_train, X_test, y_train_str, y_test_str = train_test_split(
    X, y_str,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_str
)

print("\n===== Data Summary =====")
print("Training class distribution:", y_train_str.value_counts())
print("Test class distribution:", y_test_str.value_counts())

##############################################################################
# 2) CLASSIFIER #1 (Q vs non-Q) WITH FULL RANDOMIZED SEARCH
#    - We do a pipeline with (UnderSampling -> SMOTEENN -> Scale -> XGBoost)
#    - Then we do RandomizedSearchCV over advanced XGBoost params
##############################################################################
y_train_bin1 = (y_train_str != 'Q').astype(int)  # Q => 0, NonQ => 1

pipeline_bin1 = Pipeline([
    # Option A: UnderSample the majority 'Q' to the next-largest class
    # ("undersample", RandomUnderSampler(sampling_strategy='majority', random_state=RANDOM_SEED)),

    # Option B: Partially Under-sample Q to a fixed number if you'd like to keep more Q
    #  -> Make sure your folds have at least that many Q
    ("undersample", RandomUnderSampler(
        sampling_strategy='majority',  # or {0: 10000} if feasible in your dataset
        random_state=RANDOM_SEED
    )),

    ("smote_enn", SMOTEENN(sampling_strategy='all', random_state=RANDOM_SEED)),
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_SEED
    ))
])

# Advanced XGBoost param grid
param_dist_bin1 = {
    "xgb__n_estimators": [300, 500],
    "xgb__max_depth": [4, 6, 8],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.6, 0.8],
    "xgb__colsample_bytree": [0.6, 0.8],
    "xgb__gamma": [0, 1, 5],
    "xgb__min_child_weight": [1, 3, 5],
    "xgb__reg_alpha": [0, 0.1, 1],
    "xgb__reg_lambda": [1, 5, 10],
}

random_search_bin1 = RandomizedSearchCV(
    pipeline_bin1,
    param_distributions=param_dist_bin1,
    n_iter=50,  # Increase if you can (more thorough search)
    scoring='f1',  # or 'f1_macro', 'balanced_accuracy'
    cv=3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)

print("\nFitting Classifier #1 (Q vs Non-Q) with advanced search...")
random_search_bin1.fit(X_train, y_train_bin1)
print("Best Params for Classifier #1:\n", random_search_bin1.best_params_)

best_bin1 = random_search_bin1.best_estimator_

##############################################################################
# 3) CLASSIFIER #2 (QR vs QX) - ENSEMBLE WITH FULL RANDOM SEARCH
#    - We'll do a VotingClassifier of (XGBoost + RandomForest)
#    - Another pipeline with UnderSample -> SMOTEENN -> Scale -> VotingClassifier
##############################################################################
nonq_mask_train = (y_train_str != 'Q')
X_train_nonq = X_train[nonq_mask_train]
y_train_nonq_str = y_train_str[nonq_mask_train]

# Map QR->0, QX->1
y_train_bin2 = y_train_nonq_str.map({'QR': 0, 'QX': 1}).astype(int)

xgb_bin2 = XGBClassifier(
    objective='binary:logistic',
    random_state=RANDOM_SEED
)
rf_bin2 = RandomForestClassifier(
    random_state=RANDOM_SEED
)
ensemble_bin2 = VotingClassifier(
    estimators=[("xgb", xgb_bin2), ("rf", rf_bin2)],
    voting='soft'
)

pipeline_bin2 = Pipeline([
    ("undersample", RandomUnderSampler(
        sampling_strategy='majority',
        random_state=RANDOM_SEED
    )),
    ("smote_enn", SMOTEENN(sampling_strategy='all', random_state=RANDOM_SEED)),
    ("scaler", StandardScaler()),
    ("voting", ensemble_bin2)
])

param_dist_bin2 = {
    # XGB sub-params
    "voting__xgb__n_estimators": [300, 500],
    "voting__xgb__max_depth": [4, 6],
    "voting__xgb__learning_rate": [0.01, 0.05],
    "voting__xgb__subsample": [0.6, 0.8],
    "voting__xgb__colsample_bytree": [0.6, 0.8],
    # Could also add gamma, min_child_weight, etc. if you like

    # RF sub-params
    "voting__rf__n_estimators": [100, 300],
    "voting__rf__max_depth": [5, 10],
    # Could also add min_samples_split, etc.
}

random_search_bin2 = RandomizedSearchCV(
    pipeline_bin2,
    param_distributions=param_dist_bin2,
    n_iter=20,
    scoring='f1',
    cv=3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)

print("\nFitting Classifier #2 (QR vs QX) with advanced search...")
random_search_bin2.fit(X_train_nonq, y_train_bin2)
print("Best Params for Classifier #2:\n", random_search_bin2.best_params_)

best_bin2 = random_search_bin2.best_estimator_

##############################################################################
# 4) HIERARCHICAL PREDICTION FUNCTION WITH THRESHOLDS
##############################################################################
def hierarchical_predict_with_thresholds(X_data, bin1_model, bin2_model,
                                         bin1_threshold=0.5,
                                         bin2_threshold=0.5):
    """
    1) bin1_model => Q vs nonQ (0 vs 1). Probability of '1' is prob(nonQ).
    2) Among nonQ, bin2_model => QR(0) vs QX(1). Probability of '1' is prob(QX).
    """
    prob_bin1 = bin1_model.predict_proba(X_data)[:, 1]  # prob of 'nonQ'
    pred_bin1 = (prob_bin1 >= bin1_threshold).astype(int)  # 0=Q, 1=nonQ

    final_preds = np.array([""]*len(X_data), dtype=object)
    idx_q = (pred_bin1 == 0)
    final_preds[idx_q] = "Q"

    idx_nonq = (pred_bin1 == 1)
    if np.any(idx_nonq):
        X_nonq_sub = X_data[idx_nonq]
        prob_bin2 = bin2_model.predict_proba(X_nonq_sub)[:, 1]  # prob(QX)
        pred_bin2 = (prob_bin2 >= bin2_threshold).astype(int)  # 0=QR, 1=QX
        final_preds[idx_nonq] = np.where(pred_bin2==0, "QR", "QX")

    return final_preds

label_order = ["Q", "QR", "QX"]

##############################################################################
# 5) EVALUATE ON TEST SET, TRY A SMALL GRID OF THRESHOLDS
##############################################################################
thresholds_to_try = [(0.5, 0.5), (0.4, 0.4), (0.3, 0.5), (0.5, 0.3)]
# You can systematically expand or refine these

for (th1, th2) in thresholds_to_try:
    print(f"\n=== Hierarchical Classification, Threshold=(bin1={th1}, bin2={th2}) ===")
    y_pred = hierarchical_predict_with_thresholds(
        X_test, best_bin1, best_bin2, bin1_threshold=th1, bin2_threshold=th2
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test_str,
        y_pred,
        target_names=label_order
    ))

    cm = confusion_matrix(y_test_str, y_pred, labels=label_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("\nConfusion Matrix (Row-Normalized):")
    print(cm_norm)

    # Plot normalized confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=label_order
    )
    fig, ax = plt.subplots(figsize=(6,4))
    disp.plot(
        cmap=plt.cm.Blues, 
        values_format='.2f', 
        ax=ax, 
        xticks_rotation=45
    )
    ax.set_title(f"Hierarchical CM (Normalized) - Th=({th1},{th2})")
    plt.tight_layout()
    plt.savefig(f"hierarchical_cm_{th1}_{th2}.png", dpi=300)
    plt.close()

print("\nDone. You now have a full advanced pipeline!")
