import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Read the dataset
df = pd.read_excel('C:/Users/jocel/ml/naturalDisasters.xlsx')  # Replace with your file path

# Display the first few rows of the dataset
columns_needed = [
    'Disaster Type', 'Country', 'Start Year', 'Start Month', 'Start Day',
    'End Year', 'End Month', 'End Day', 'Total Deaths', 'No. Injured',
    'Total Affected', 'Disaster Group', 'Disaster Subgroup'
]
df = df[columns_needed]

# Fill missing values for numeric columns and assign placeholder for missing date parts
df = df.fillna({
    'Total Deaths': 0,
    'No. Injured': 0,
    'Total Affected': 0,
    'Start Month': -1,
    'Start Day': -1,
    'End Month': -1,
    'End Day': -1
})

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
df.drop(columns=['Start Year', 'Start Month', 'Start Day',
                 'End Year', 'End Month', 'End Day',
                 'Start Date', 'End Date'], inplace=True)

# Create Year and Month columns from Start Date
label_encoder = LabelEncoder()
df['Disaster Type'] = label_encoder.fit_transform(df['Disaster Type'])
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping for Disaster Type:", mapping)

df = pd.get_dummies(df, columns=['Country', 'Disaster Group', 'Disaster Subgroup'], drop_first=True)

# Display the first few rows of the processed dataset
X = df.drop(columns=['Disaster Type'])
y = df['Disaster Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Random Forest ---
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

print("\nRandom Forest Hyperparameter Results (Accuracy):")
for n_estimators in rf_param_grid['n_estimators']:
    for max_depth in rf_param_grid['max_depth']:
        for min_samples_split in rf_param_grid['min_samples_split']:
            for min_samples_leaf in rf_param_grid['min_samples_leaf']:
                for bootstrap in rf_param_grid['bootstrap']:
                    rf_model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        bootstrap=bootstrap,
                        random_state=42
                    )
                    rf_model.fit(X_train, y_train)
                    pred_rf = rf_model.predict(X_test)
                    acc_rf = accuracy_score(y_test, pred_rf)
                    print(f"RF: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, bootstrap={bootstrap} => Accuracy: {acc_rf:.4f}")

# --- XGBoost ---
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1],
    'colsample_bytree': [0.5, 0.7, 1]
}

print("\nXGBoost Hyperparameter Results (Accuracy):")
for n_estimators in xgb_param_grid['n_estimators']:
    for max_depth in xgb_param_grid['max_depth']:
        for learning_rate in xgb_param_grid['learning_rate']:
            for subsample in xgb_param_grid['subsample']:
                for colsample_bytree in xgb_param_grid['colsample_bytree']:
                    xgb_model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        eval_metric='mlogloss',
                        random_state=42
                    )
                    xgb_model.fit(X_train, y_train)
                    pred_xgb = xgb_model.predict(X_test)
                    acc_xgb = accuracy_score(y_test, pred_xgb)
                    print(f"XGB: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}, colsample_bytree={colsample_bytree} => Accuracy: {acc_xgb:.4f}")

# --- CatBoost ---
cat_param_grid = {
    'iterations': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128]
}

# Display the first few rows of the processed dataset
print("\nCatBoost Hyperparameter Results (Accuracy):")
for iterations in cat_param_grid['iterations']:
    for learning_rate in cat_param_grid['learning_rate']:
        for depth in cat_param_grid['depth']:
            for l2_leaf_reg in cat_param_grid['l2_leaf_reg']:
                for border_count in cat_param_grid['border_count']:
                    cat_model = CatBoostClassifier(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        l2_leaf_reg=l2_leaf_reg,
                        border_count=border_count,
                        loss_function='MultiClass',
                        random_seed=42,
                        verbose=False
                    )
                    cat_model.fit(X_train, y_train)
                    pred_cat = cat_model.predict(X_test)
                    acc_cat = accuracy_score(y_test, pred_cat)
                    print(f"CAT: iterations={iterations}, learning_rate={learning_rate}, depth={depth}, l2_leaf_reg={l2_leaf_reg}, border_count={border_count} => Accuracy: {acc_cat:.4f}")


