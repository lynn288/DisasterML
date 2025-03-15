import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pickle

# =============================================================================
# 4. PREPARING DATA FOR MODELING
# =============================================================================
# Load the processed CSV file
df = pd.read_csv('processedNaturalDisasters.csv')

# Define features (X) and target (y)
X = df.drop(columns=['Disaster Type'])
y = df['Disaster Type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# 5. MODEL TRAINING
# =============================================================================
# 5a. Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5b. XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# 5c. CatBoost Classifier
cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, loss_function='MultiClass',
                               random_seed=42, verbose=False)
cat_model.fit(X_train, y_train)

# 5e. Ensemble Model: Voting Classifier (Soft Voting)
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('cat', cat_model)
], voting='soft')
ensemble_model.fit(X_train, y_train)

# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================
# Get predictions from each model
y_pred_ensemble = ensemble_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_cat = cat_model.predict(X_test)

# Print evaluation metrics for each model
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("\nEnsemble Classification Report:\n", classification_report(y_test, y_pred_ensemble, zero_division=0))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb, zero_division=0))

print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
print("CatBoost Classification Report:\n", classification_report(y_test, y_pred_cat, zero_division=0))

# Combine all confusion matrices in one figure (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Ensemble Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Ensemble Model Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# XGBoost Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('XGBoost Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# CatBoost Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_cat), annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('CatBoost Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# =============================================================================
# 7. SAVE MODELS AND SCALER
# =============================================================================
with open("ensemble_model.pkl", "wb") as file:
    pickle.dump(ensemble_model, file)

with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

with open("xgb_model.pkl", "wb") as file:
    pickle.dump(xgb_model, file)

with open("catboost_model.pkl", "wb") as file:
    pickle.dump(cat_model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# =============================================================================
# 8. DISPLAY THE SAVED PICKLE FILES (all at once)
# =============================================================================
def display_pickle(file_path, description):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    print(f"{description}:")
    print(obj)
    print("\n" + "="*80 + "\n")

display_pickle("ensemble_model.pkl", "Ensemble Model")
display_pickle("rf_model.pkl", "Random Forest Model")
display_pickle("xgb_model.pkl", "XGBoost Model")
display_pickle("catboost_model.pkl", "CatBoost Model")
display_pickle("scaler.pkl", "Scaler")
