import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
#pip install seaborn
#pip install xgboost
import seaborn as sns
from xgboost import XGBClassifier

# =============================================================================
# 1. DATA LOADING & INITIAL PREPROCESSING
# =============================================================================
# Load the dataset
df = pd.read_excel('C:/Users/jocel/ml/naturalDisasters.xlsx')  # Replace with your file path

# Select only the required columns
columns_needed = [
    'Disaster Type', 'Country', 'Start Year', 'Start Month', 'Start Day',
    'End Year', 'End Month', 'End Day', 'Total Deaths', 'No. Injured',
    'Total Affected', 'Disaster Group', 'Disaster Subgroup'
]
df = df[columns_needed]

# Handle missing values for numeric columns and assign placeholder (-1) for missing month/day values
df = df.fillna({
    'Total Deaths': 0,
    'No. Injured': 0,
    'Total Affected': 0,
    'Start Month': -1,
    'Start Day': -1,
    'End Month': -1,
    'End Day': -1
})

# =============================================================================
# 2. DATE CONVERSION & FEATURE ENGINEERING
# =============================================================================
# Convert Start Date
start_date_df = df[['Start Year', 'Start Month', 'Start Day']].replace(-1, np.nan) \
               .rename(columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'})
df['Start Date'] = pd.to_datetime(start_date_df)

# Convert End Date
end_date_df = df[['End Year', 'End Month', 'End Day']].replace(-1, np.nan) \
             .rename(columns={'End Year': 'year', 'End Month': 'month', 'End Day': 'day'})
df['End Date'] = pd.to_datetime(end_date_df)

# Create a new feature: Duration (in days)
df['Duration'] = (df['End Date'] - df['Start Date']).dt.days.fillna(0)

# Drop original date columns (they've been processed)
df.drop(columns=['Start Year', 'Start Month', 'Start Day',
                 'End Year', 'End Month', 'End Day',
                 'Start Date', 'End Date'], inplace=True)

# =============================================================================
# 3. ENCODING CATEGORICAL VARIABLES
# =============================================================================
# Label Encode the target: 'Disaster Type'
label_encoder = LabelEncoder()
df['Disaster Type'] = label_encoder.fit_transform(df['Disaster Type'])

# Print the mapping for target labels for reference
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping for Disaster Type:", mapping)

# 8. One-hot encode Country, Group, Subgroup only
df = pd.get_dummies(df, columns=['Country', 'Disaster Group', 'Disaster Subgroup'], drop_first=True)

# =============================================================================
# 4. PREPARING DATA FOR MODELING
# =============================================================================
# Define features (X) and target (y)
X = df.drop(columns=['Disaster Type'])
y = df['Disaster Type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Or should we use 0.3?

# Scale numerical features 
#!! Normalise dataset one of her lectures said must remember to do this to optimise accuracy..
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# 5. MODEL TRAINING
# =============================================================================
# --------------------------
# 5a. Random Forest Classifier
# --------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --------------------------
# 5b. Gradient Boosting Classifier (XGBoost)
# --------------------------
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# --------------------------
# 5c. Ensemble Model: Voting Classifier (Soft Voting)
# --------------------------
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')
ensemble_model.fit(X_train, y_train)

# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================
# Make predictions using the ensemble model
y_pred = ensemble_model.predict(X_test)

# Evaluate model performance
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
print("\nEnsemble Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Plot the Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Ensemble Model Confusion Matrix')
plt.show()

# Evaluate Random Forest alone
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred, zero_division=0))
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Evaluate XGBoost alone
xgb_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred, zero_division=0))
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.show()

