import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data_df    = pd.read_csv('data.csv')
predict_df = pd.read_csv('predict.csv')

# Feature engineering
def add_features(df):
    df = df.copy()
    df['run_diff']       = df['R'] - df['RA']
    df['run_ratio']      = df['R'] / (df['RA'] + 1)
    df['OBP']            = (df['H'] + df['BB']) / (df['AB'] + df['BB'] + 1)
    df['SLG']            = (df['H'] + df['2B'] + 2*df['3B'] + 3*df['HR']) / (df['AB'] + 1)
    df['OPS']            = df['OBP'] + df['SLG']
    df['WHIP']           = (df['BBA'] + df['HA']) / (df['IPouts'] / 3 + 1)
    df['K_BB_ratio']     = df['SOA'] / (df['BBA'] + 1)
    df['pyth_wins']      = (df['R']**1.84 / (df['R']**1.84 + df['RA']**1.84)) * df['G']
    IP = df['IPouts'] / 3 + 0.001
    df['FIP']            = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / IP + 3.2
    df['def_efficiency'] = 1 - (df['HA'] - df['HRA']) / (df['BBA'] + df['HA'] - df['HRA'] + 1)
    return df

data_df    = add_features(data_df)
predict_df = add_features(predict_df)

# Feature selection
drop_cols    = ['yearID', 'teamID', 'year_label', 'decade_label', 'win_bins', 'W', 'ID']
feature_cols = [c for c in data_df.columns if c not in drop_cols and c in predict_df.columns]

# Scaling: StandardScaler on non-binary features only
one_hot_cols = [c for c in feature_cols if c.startswith(('era_', 'decade_'))]
other_cols   = [c for c in feature_cols if c not in one_hot_cols]

X         = data_df[feature_cols]
y         = data_df['W']
X_predict = predict_df[feature_cols]

scaler = StandardScaler()
X_scaled              = X.copy()
X_scaled[other_cols]  = scaler.fit_transform(X[other_cols])
X_pred_scaled         = X_predict.copy()
X_pred_scaled[other_cols] = scaler.transform(X_predict[other_cols])

# ── Evaluate with multiple metrics (5-fold CV) ────────────────────────────
print("=== 5-Fold Cross-Validation Evaluation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []

for train_idx, val_idx in kf.split(X_scaled):
    m = Lasso(alpha=0.008, max_iter=10000)
    m.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])
    y_true_all.extend(y.iloc[val_idx])
    y_pred_all.extend(m.predict(X_scaled.iloc[val_idx]))

y_true   = np.array(y_true_all)
y_pred_r = np.round(np.array(y_pred_all))

print(f"MAE:  {mean_absolute_error(y_true, y_pred_r):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_r)):.4f}")
print(f"R²:   {r2_score(y_true, y_pred_r):.4f}")
print(f"Bias: {np.mean(y_pred_r - y_true):.4f}  (positive = over-predict)")
print()
print("Prediction error distribution:")
for t in [1, 2, 3, 5, 10]:
    pct = np.mean(np.abs(y_pred_r - y_true) <= t) * 100
    print(f"  Within ±{t:2d} wins: {pct:.1f}%")

# ── Train on full data and generate submission ────────────────────────────
print("\n=== Generating Submission ===")
model = Lasso(alpha=0.008, max_iter=10000)
model.fit(X_scaled, y)
preds = model.predict(X_pred_scaled)

submission = pd.DataFrame({
    'ID': predict_df['ID'],
    'W':  np.round(preds).astype(int)
})
submission.to_csv('submission.csv', index=False)
print(f"Saved {len(submission)} predictions")
print(f"W range: {submission['W'].min()} - {submission['W'].max()}")
