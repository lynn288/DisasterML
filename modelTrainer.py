import time
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pickle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Load the processed CSV file (with binary target)
df = pd.read_csv('processedNaturalDisasters.csv')

# NEW: Create a date column using 'Year' and 'Month'
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
# NEW: Sort by date to ensure chronological order
df = df.sort_values(by='Date')

# Replace random splitting with a date-based split:
cutoff_date = pd.Timestamp('2016-01-01')
train_df = df[df['Date'] < cutoff_date]
test_df = df[df['Date'] >= cutoff_date]

# Select only numeric columns (exclude string columns) for modeling
numeric_train = train_df.select_dtypes(exclude=['object'])
numeric_test = test_df.select_dtypes(exclude=['object'])

X_train = numeric_train.drop(columns=['Disaster Occurred', 'Date'])
y_train = numeric_train['Disaster Occurred']

X_test = numeric_test.drop(columns=['Disaster Occurred', 'Date'])
y_test = numeric_test['Disaster Occurred']

# Function to plot feature importances
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier with hyperparameters as provided
start = time.time()
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30,
                                  min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)  
rf_model.fit(X_train_scaled, y_train)
rf_training_time = time.time() - start


# XGBoost Classifier with hyperparameters as provided
start = time.time()
xgb_model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.2,
                          subsample=0.5, colsample_bytree=0.5, use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42)  
xgb_model.fit(X_train_scaled, y_train)
xgb_training_time = time.time() - start


# CatBoost Classifier with hyperparameters as provided
start = time.time()
cat_model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=8, l2_leaf_reg=1,
                               border_count=32, loss_function='MultiClass',
                               random_seed=42, verbose=False)
cat_model.fit(X_train_scaled, y_train)
cat_training_time = time.time() - start

# Ensemble Model: Voting Classifier (Soft Voting)
start = time.time()
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('cat', cat_model)
], voting='soft')
ensemble_model.fit(X_train_scaled, y_train)
ensemble_training_time = time.time() - start

# Get predictions from each model (test set)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_cat = cat_model.predict(X_test_scaled)

# Get predictions from each model (training set) for overfitting check
y_pred_ensemble_train = ensemble_model.predict(X_train_scaled)
y_pred_rf_train = rf_model.predict(X_train_scaled)
y_pred_xgb_train = xgb_model.predict(X_train_scaled)
y_pred_cat_train = cat_model.predict(X_train_scaled)

# Calculate accuracies for both test and train sets
test_acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
train_acc_ensemble = accuracy_score(y_train, y_pred_ensemble_train)

# Calculate accuracies for both test and train sets for Random Forest, XGBoost, and CatBoost
# Random Forest
test_acc_rf = accuracy_score(y_test, y_pred_rf)
train_acc_rf = accuracy_score(y_train, y_pred_rf_train)

# XGBoost
test_acc_xgb = accuracy_score(y_test, y_pred_xgb)
train_acc_xgb = accuracy_score(y_train, y_pred_xgb_train)

# CatBoost
test_acc_cat = accuracy_score(y_test, y_pred_cat)
train_acc_cat = accuracy_score(y_train, y_pred_cat_train)

# Display the results
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Ensemble ROC-AUC:", roc_auc_score(y_test, ensemble_model.predict_proba(X_test_scaled)[:,1]))
print("\nEnsemble Classification Report:\n", classification_report(y_test, y_pred_ensemble, zero_division=0))
print("Overfitting Check:")
print(f"   Train Accuracy: {train_acc_ensemble:.4f}")
print(f"   Test Accuracy:  {test_acc_ensemble:.4f}")
print(f"   Gap (Train-Test): {train_acc_ensemble - test_acc_ensemble:.4f}")
print(f"   Overfitting Ratio: {train_acc_ensemble/test_acc_ensemble:.4f}\n")

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
print("Overfitting Check:")
print(f"   Train Accuracy: {train_acc_rf:.4f}")
print(f"   Test Accuracy:  {test_acc_rf:.4f}")
print(f"   Gap (Train-Test): {train_acc_rf - test_acc_rf:.4f}")
print(f"   Overfitting Ratio: {train_acc_rf/test_acc_rf:.4f}\n")

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb, zero_division=0))
print("Overfitting Check:")
print(f"   Train Accuracy: {train_acc_xgb:.4f}")
print(f"   Test Accuracy:  {test_acc_xgb:.4f}")
print(f"   Gap (Train-Test): {train_acc_xgb - test_acc_xgb:.4f}")
print(f"   Overfitting Ratio: {train_acc_xgb/test_acc_xgb:.4f}\n")

print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
print("CatBoost Classification Report:\n", classification_report(y_test, y_pred_cat, zero_division=0))
print("Overfitting Check:")
print(f"   Train Accuracy: {train_acc_cat:.4f}")
print(f"   Test Accuracy:  {test_acc_cat:.4f}")
print(f"   Gap (Train-Test): {train_acc_cat - test_acc_cat:.4f}")
print(f"   Overfitting Ratio: {train_acc_cat/test_acc_cat:.4f}\n")

# Summary table for all models
print("\nOverfitting Summary Table:")
models = ["Ensemble", "Random Forest", "XGBoost", "CatBoost"]
train_accs = [train_acc_ensemble, train_acc_rf, train_acc_xgb, train_acc_cat]
test_accs = [test_acc_ensemble, test_acc_rf, test_acc_xgb, test_acc_cat]
gaps = [train - test for train, test in zip(train_accs, test_accs)]
ratios = [train/test for train, test in zip(train_accs, test_accs)]

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Model': models,
    'Train Accuracy': train_accs,
    'Test Accuracy': test_accs,
    'Gap (Overfitting)': gaps,
    'Overfitting Ratio': ratios
})
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Add extra new line to make it readable
print()
print()

# Use existing predictions
model_preds = {
    "Ensemble": y_pred_ensemble,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "CatBoost": y_pred_cat
}

# Reference to models
models_dict = {
    "Ensemble": ensemble_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "CatBoost": cat_model
}

# Initialize lists to store metrics
precisions = []
recalls = []
f1s = []
test_times = []
train_times = []
memory_usages = []

# Measure memory usage
memory_usages = []

# Measure memory usage for each model
for name in models:
    tracemalloc.start()  # Start memory tracking
    
    _ = models_dict[name].predict(X_test_scaled)  # Run inference
    
    _, peak_memory = tracemalloc.get_traced_memory()  # Get peak memory usage
    tracemalloc.stop()  # Stop tracking
    
    memory_usages.append(peak_memory / (1024 * 1024)) 

# Store training times when models were first trained
train_times_dict = {
    "Ensemble": ensemble_training_time,
    "Random Forest": rf_training_time,
    "XGBoost": xgb_training_time,
    "CatBoost": cat_training_time
}

train_times = [train_times_dict[name] for name in models]

# Measure test time and compute metrics
for name in models:
    # Time prediction
    start = time.time()
    _ = models_dict[name].predict(X_test_scaled)
    end = time.time()
    test_times.append(end - start)

    # Metrics from existing predictions
    precisions.append(precision_score(y_test, model_preds[name], zero_division=0))
    recalls.append(recall_score(y_test, model_preds[name], zero_division=0))
    f1s.append(f1_score(y_test, model_preds[name], zero_division=0))

# Final summary
performance_df = pd.DataFrame({
    "Model": models,
    "Test Accuracy": test_accs,
    "Precision": precisions,
    "Recall": recalls,
    "F1 Score": f1s,
    "Train Time (s)": train_times,
    "Test Time (s)": test_times,
    "Memory Usage (MB)": memory_usages
})

print("Extended Performance Summary:")
print(performance_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# Combine all confusion matrices in one figure (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot confusion matrices
# Ensemble Model
sns.heatmap(confusion_matrix(y_test, y_pred_ensemble), annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Ensemble Model Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# XGBoost
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('XGBoost Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# CatBoost
sns.heatmap(confusion_matrix(y_test, y_pred_cat), annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('CatBoost Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

# Adjust layout
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.close()

# Plot train vs test accuracy comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

# Plot bars
plt.bar(x - width/2, train_accs, width, label='Train Accuracy')
plt.bar(x + width/2, test_accs, width, label='Test Accuracy')

# Add labels, title, and legend
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy Comparison')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.savefig('train_vs_test_accuracy.png')
plt.close()

# Save the models and the scaler
# Ensemble Model
with open("ensemble_model.pkl", "wb") as file:
    pickle.dump(ensemble_model, file)
    
# Random Forest Model
with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

# XGBoost Model
with open("xgb_model.pkl", "wb") as file:
    pickle.dump(xgb_model, file)
with open("X_test_scaled.pkl", "wb") as f:
    pickle.dump(X_test_scaled, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

# CatBoost Model
with open("catboost_model.pkl", "wb") as file:
    pickle.dump(cat_model, file)

# Scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
       
# Display the saved models
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
