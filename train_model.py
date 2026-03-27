"""
train_model.py
Loads trades.csv, engineers features, trains an XGBoost win/loss classifier,
prints evaluation metrics, and saves model.pkl + importance.png.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("trades.csv")
print(f"Loaded {len(df)} trades  |  win rate: {(df['result']=='win').mean():.1%}\n")

# ── Feature engineering ───────────────────────────────────────────────────────
# 1. Bias–direction alignment flag
df["bias_aligned"] = (
    ((df["htf_bias"] == "bullish") & (df["trade_direction"] == "long")) |
    ((df["htf_bias"] == "bearish") & (df["trade_direction"] == "short"))
).astype(int)

# 2. RSI zone
df["rsi_zone"] = pd.cut(
    df["rsi_at_entry"],
    bins=[0, 35, 45, 55, 65, 100],
    labels=["oversold", "low_neutral", "mid_neutral", "high_neutral", "overbought"]
)

# 3. Volume tier
df["vol_tier"] = pd.cut(
    df["volume_ratio"],
    bins=[0, 0.8, 1.0, 1.2, 1.5, 99],
    labels=["very_low", "low", "normal", "high", "very_high"]
)

# 4. EMA direction aligned with trade
df["ema_aligned"] = (
    ((df["ema_diff"] > 0) & (df["trade_direction"] == "long")) |
    ((df["ema_diff"] < 0) & (df["trade_direction"] == "short"))
).astype(int)

# 5. Interaction: bias_aligned × session quality
session_quality = {"london": 2, "newyork": 2, "asia": 1, "overnight": 0}
df["session_quality"] = df["session"].map(session_quality)
df["bias_x_session"] = df["bias_aligned"] * df["session_quality"]

# 6. RSI distance from 50
df["rsi_dist_50"] = (df["rsi_at_entry"] - 50).abs()

# 7. EMA magnitude (normalised)
df["ema_abs"] = df["ema_diff"].abs()

# ── Encode categoricals ───────────────────────────────────────────────────────
cat_cols = ["timeframe", "session", "htf_bias", "trade_direction",
            "rsi_zone", "vol_tier"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ── Feature matrix ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "timeframe_enc", "session_enc", "htf_bias_enc", "trade_direction_enc",
    "rsi_at_entry", "ema_diff", "volume_ratio", "sl_distance_points",
    "bias_aligned", "ema_aligned", "session_quality",
    "bias_x_session", "rsi_dist_50", "ema_abs",
    "rsi_zone_enc", "vol_tier_enc",
]

X = df[FEATURE_COLS]
y = (df["result"] == "win").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Train XGBoost ─────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("=" * 55)
print("  MODEL EVALUATION")
print("=" * 55)
print(f"  Accuracy   (test)      : {accuracy:.3f}")
print(f"  Precision  (test)      : {precision:.3f}")
print(f"  Recall     (test)      : {recall:.3f}")
print(f"  CV Accuracy (5-fold)   : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["loss", "win"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}\n")

# ── Feature importance ────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature":   FEATURE_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("Top 10 Features:")
print(importance_df.head(10).to_string(index=False))

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump({"model": model, "label_encoders": label_encoders,
             "feature_cols": FEATURE_COLS}, "model.pkl")
print("\nSaved → model.pkl")

# ── Feature importance chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(
    data=importance_df,
    x="importance", y="feature",
    palette="viridis", ax=ax
)
ax.set_title("XGBoost Feature Importance — IFVG Trade Model",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)

# Annotate accuracy on chart
ax.text(
    0.98, 0.02,
    f"Test Accuracy: {accuracy:.1%}  |  CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=9, color="gray",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8)
)

plt.tight_layout()
plt.savefig("importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → importance.png")
