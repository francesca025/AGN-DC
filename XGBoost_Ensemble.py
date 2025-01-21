import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn / Imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline  # from imblearn, not sklearn

# Classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_SEED = 42

##############################################################################
# 1) LOAD AND PREPARE DATA
##############################################################################
df = pd.read_csv('gr_features_by_object.csv', index_col=False)
constructed_df = pd.read_csv('new_features.csv', index_col=False)
merged = constructed_df.merge(df, on='name', how='inner')

valid_types = ['Q', 'QR', 'QX']
data_df = merged[merged['type'].isin(valid_types)].copy()
data_df.dropna(inplace=True)

# Example additional features
data_df['variance_ratio'] = (
    data_df['Meanvariance_g'] / (data_df['Meanvariance_r'] + 1e-8)
)
data_df['mean_band_ratio'] = (
    data_df['Mean_g'] / (data_df['Mean_r'] + 1e-8)
)

X_all = data_df.drop(columns=['name', 'type'])
y_str_all = data_df['type']

X_train, X_test, y_train_str, y_test_str = train_test_split(
    X_all, y_str_all,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_str_all
)

print("\nTrain distribution:\n", y_train_str.value_counts())
print("Test distribution:\n", y_test_str.value_counts())

##############################################################################
# 2) CLASSIFIER #1: (Q vs. Non-Q) Pipeline
##############################################################################
y_train_bin1 = (y_train_str != 'Q').astype(int)  # 0 => Q, 1 => Non-Q

# Here we can do a single pipeline that ends in XGB, so we just call .fit()
pipeline_bin1 = Pipeline([
    ("undersample", RandomUnderSampler(sampling_strategy='majority', random_state=RANDOM_SEED)),
    ("smote_enn", SMOTEENN(sampling_strategy='all', random_state=RANDOM_SEED)),
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_SEED,
        # Example best params:
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.8,
        gamma=0,
        min_child_weight=3,
        reg_alpha=1,
        reg_lambda=10
    ))
])

print("\nFitting Classifier #1 (Q vs. Non-Q)...")
pipeline_bin1.fit(X_train, y_train_bin1)

##############################################################################
# 3) Build Data for Classifier #2: add "prob_nonq_from_cls1"
##############################################################################
nonq_mask_train = (y_train_str != 'Q')
X_train_nonq = X_train[nonq_mask_train].copy()  # copies only non-Q
y_train_nonq_str = y_train_str[nonq_mask_train] # QR or QX

# Probability from pipeline_bin1 (the chance of being NonQ)
prob_bin1_train = pipeline_bin1.predict_proba(X_train_nonq)[:, 1]
X_train_nonq['prob_nonq_from_cls1'] = prob_bin1_train

# Convert QR->0, QX->1
y_train_bin2 = y_train_nonq_str.map({'QR':0, 'QX':1}).astype(int)

##############################################################################
# 4) Resample for Classifier #2 (QR vs QX), then scale, then fit Weighted/Calibrated classifier
##############################################################################
# (a) Make a pipeline that ends with SMOTEENN, so we can .fit_resample()
resampler2 = Pipeline([
    ("under", RandomUnderSampler(sampling_strategy='majority', random_state=RANDOM_SEED)),
    ("smote_enn", SMOTEENN(sampling_strategy='all', random_state=RANDOM_SEED))
])
print("\nResampling data for Classifier #2 (QR vs QX)...")
X_train_nonq_res, y_train_bin2_res = resampler2.fit_resample(X_train_nonq, y_train_bin2)

# (b) Scale
scaler2 = StandardScaler()
X_train_nonq_res_np = scaler2.fit_transform(X_train_nonq_res)

# (c) Weighted approach => e.g. errors on QR=label0 cost double
sample_weight_bin2 = np.where(y_train_bin2_res == 0, 2.0, 1.0)

# (d) Voting classifier (XGB + RF)
xgb_bin2 = XGBClassifier(
    objective='binary:logistic',
    random_state=RANDOM_SEED,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.6,
    colsample_bytree=0.6
)
rf_bin2 = RandomForestClassifier(
    random_state=RANDOM_SEED,
    n_estimators=300,
    max_depth=5
)
voting_model_bin2 = VotingClassifier(
    estimators=[('xgb', xgb_bin2), ('rf', rf_bin2)],
    voting='soft'
)

print("\nFitting VotingClassifier for Classifier #2 w/ sample_weight...")
voting_model_bin2.fit(X_train_nonq_res_np, y_train_bin2_res, sample_weight=sample_weight_bin2)

# (e) Calibrate for better probability estimates
calibrated_bin2 = CalibratedClassifierCV(
    base_estimator=voting_model_bin2,
    method='sigmoid',
    cv=3
)
print("Calibrating VotingClassifier #2 probabilities...")
calibrated_bin2.fit(X_train_nonq_res_np, y_train_bin2_res, sample_weight=sample_weight_bin2)

# We'll store the columns from the resampled DataFrame so we can align test data
# (including the new 'prob_nonq_from_cls1' column).
used_cols_bin2 = X_train_nonq_res.columns.tolist()

##############################################################################
# 5) Hierarchical prediction function
##############################################################################
def hierarchical_predict(
    X_data,
    bin1_pipeline,      # pipeline for Q vs NonQ
    bin2_calibrated,    # calibrated classifier #2
    scaler_for_bin2,    # the StandardScaler we used for bin2
    used_cols,          # columns in X_train_nonq_res
    threshold_bin1=0.5,
    threshold_bin2=0.5
):
    """
    1) Probability of Non-Q from bin1 => compare threshold_bin1 => Q or NonQ
    2) If NonQ => create the 'prob_nonq_from_cls1' column,
       realign columns to used_cols,
       scale with scaler_for_bin2,
       get prob(QX) from bin2_calibrated => threshold_bin2 => QR or QX
    """
    prob_bin1 = bin1_pipeline.predict_proba(X_data)[:, 1]  # prob(NonQ)
    pred_bin1 = (prob_bin1 >= threshold_bin1).astype(int)  # 0=Q, 1=NonQ

    final_preds = np.array([""]*len(X_data), dtype=object)
    # Mark Q
    idx_q = (pred_bin1 == 0)
    final_preds[idx_q] = "Q"

    # Among NonQ => classifier #2
    idx_nonq = (pred_bin1 == 1)
    if np.any(idx_nonq):
        X_nonq_sub = X_data[idx_nonq].copy()
        # Add the meta-feature
        prob_nonq_sub = prob_bin1[idx_nonq]
        X_nonq_sub['prob_nonq_from_cls1'] = prob_nonq_sub

        # Align columns to used_cols
        X_nonq_aligned = pd.DataFrame(columns=used_cols, index=X_nonq_sub.index)
        for c in used_cols:
            if c in X_nonq_sub.columns:
                X_nonq_aligned[c] = X_nonq_sub[c]
            else:
                X_nonq_aligned[c] = 0.0

        # Scale
        X_nonq_scaled = scaler_for_bin2.transform(X_nonq_aligned)

        # Predict prob(QX)
        prob_bin2 = bin2_calibrated.predict_proba(X_nonq_scaled)[:, 1]
        pred_bin2 = (prob_bin2 >= threshold_bin2).astype(int)  # 0 => QR, 1 => QX

        final_preds[idx_nonq] = np.where(pred_bin2 == 0, "QR", "QX")

    return final_preds

##############################################################################
# 6) Evaluate on the test set at different thresholds
##############################################################################
label_order = ["Q","QR","QX"]
threshold_pairs = [
    (0.5, 0.5),
    (0.4, 0.4),
    (0.3, 0.5),
    (0.5, 0.3),
]

for (th1, th2) in threshold_pairs:
    print(f"\n=== Hierarchical Classification, (bin1={th1}, bin2={th2}) ===")
    y_pred = hierarchical_predict(
        X_test,
        pipeline_bin1,
        calibrated_bin2,
        scaler2,
        used_cols_bin2,
        threshold_bin1=th1,
        threshold_bin2=th2
    )

    print("\nClassification Report:")
    print(classification_report(y_test_str, y_pred, target_names=label_order))

    cm = confusion_matrix(y_test_str, y_pred, labels=label_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("\nConfusion Matrix (Row-Normalized):")
    print(cm_norm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=label_order)
    fig, ax = plt.subplots(figsize=(6,4))
    disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax, xticks_rotation=45)
    ax.set_title(f"Hierarchical CM (Normalized) - (bin1={th1}, bin2={th2})")
    plt.tight_layout()
    plt.savefig(f"hierarchical_cm_bin1_{th1}_bin2_{th2}.png", dpi=300)
    plt.close()

print("\nAll done with the corrected hierarchical classification code!")
