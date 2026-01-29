# Heart Disease Prediction using Logistic Regression and KNN
# =======================
# Step 1: Import Libraries
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)

# =======================
# Step 2: Load Dataset
# (EDITED LINE – VERY IMPORTANT)
# =======================
df = pd.read_csv("heart.csv")   # ✅ CHANGED FROM HARD-CODED PATH
print("Dataset Shape:", df.shape)
print(df.head())

# =======================
# Step 3: Data Preprocessing
# =======================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# Step 4: Logistic Regression
# =======================
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

lr_acc = accuracy_score(y_test, y_pred_lr)

print("\nLogistic Regression Accuracy:", lr_acc)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# =======================
# Logistic Regression Confusion Matrix (Clean %)
# =======================
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_lr_percent = cm_lr / cm_lr.sum() * 100

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_lr_percent,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)
plt.title("Confusion Matrix – Logistic Regression (%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =======================
# ROC Curve – Logistic Regression
# =======================
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# Feature Importance (Logistic Regression)
# =======================
feature_importance = pd.Series(
    lr_model.coef_[0],
    index=X.columns
).sort_values()

plt.figure(figsize=(8, 6))
feature_importance.plot(kind="barh")
plt.title("Feature Importance – Logistic Regression")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# =======================
# Step 5: KNN Model
# =======================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)

print("\nKNN Accuracy (K=5):", knn_acc)
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

# =======================
# KNN Confusion Matrix
# =======================
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_knn_percent = cm_knn / cm_knn.sum() * 100

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_knn_percent,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)
plt.title("Confusion Matrix – KNN (%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =======================
# Step 6: Accuracy Comparison Bar Chart
# =======================
models = ["Logistic Regression", "KNN"]
accuracies = [lr_acc, knn_acc]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.show()

# =======================
# Step 7: K Value vs Accuracy (KNN)
# =======================
accuracy_scores = []
k_values = range(1, 21)

for k in k_values:
    temp_knn = KNeighborsClassifier(n_neighbors=k)
    temp_knn.fit(X_train, y_train)
    pred_k = temp_knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, pred_k))

plt.figure(figsize=(7, 5))
plt.plot(k_values, accuracy_scores, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy (KNN)")
plt.grid(True)
plt.tight_layout()
plt.show()

best_k = k_values[np.argmax(accuracy_scores)]
print(f"\nBest K value: {best_k}")

