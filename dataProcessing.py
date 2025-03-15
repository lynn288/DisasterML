import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def correlation_feature_selection(df, target_col, corr_threshold=0.9):
    """
    Drop features that are highly correlated with each other.
    
    For each pair of features with correlation above corr_threshold,
    the feature with lower absolute correlation with the target is dropped.
    
    Parameters:
        df (DataFrame): Input dataframe containing features and target.
        target_col (str): Name of the target column.
        corr_threshold (float): Correlation threshold for dropping features.
        
    Returns:
        List of columns to drop.
    """
    # Select only numeric columns for correlation computation
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]
    corr_matrix = numeric_df.corr().abs()
    target_corr = corr_matrix[target_col]
    
    # Get list of feature columns (exclude target)
    features = [col for col in numeric_df.columns if col != target_col]
    
    drop_cols = set()
    # Compare each pair of features
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            col1, col2 = features[i], features[j]
            if corr_matrix.loc[col1, col2] > corr_threshold:
                # Drop the feature with the lower correlation to the target
                if target_corr[col1] >= target_corr[col2]:
                    drop_cols.add(col2)
                else:
                    drop_cols.add(col1)
    return list(drop_cols)

def load_and_preprocess_data(input_file, output_file, corr_threshold=0.9):
    # -----------------------------
    # 1. DATA LOADING & INITIAL PREPROCESSING
    # -----------------------------
    df = pd.read_excel(input_file)
    columns_needed = [
        'Disaster Type', 'Country', 'Start Year', 'Start Month', 'Start Day',
        'End Year', 'End Month', 'End Day', 'Total Deaths', 'No. Injured',
        'Total Affected', 'Disaster Group', 'Disaster Subgroup'
    ]
    df = df[columns_needed]
    
    df = df.fillna({
        'Total Deaths': 0,
        'No. Injured': 0,
        'Total Affected': 0,
        'Start Month': -1,
        'Start Day': -1,
        'End Month': -1,
        'End Day': -1
    })
    
    # -----------------------------
    # 2. DATE CONVERSION & FEATURE ENGINEERING
    # -----------------------------
    # Convert Start Date
    start_date_df = df[['Start Year', 'Start Month', 'Start Day']].replace(-1, np.nan)\
                    .rename(columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'})
    df['Start Date'] = pd.to_datetime(start_date_df)
    
    # Convert End Date
    end_date_df = df[['End Year', 'End Month', 'End Day']].replace(-1, np.nan)\
                  .rename(columns={'End Year': 'year', 'End Month': 'month', 'End Day': 'day'})
    df['End Date'] = pd.to_datetime(end_date_df)
    
    # Create Duration feature (in days)
    df['Duration'] = (df['End Date'] - df['Start Date']).dt.days.fillna(0)
    
    # Drop original date columns
    df.drop(columns=['Start Year', 'Start Month', 'Start Day',
                     'End Year', 'End Month', 'End Day',
                     'Start Date', 'End Date'], inplace=True)
    
    # -----------------------------
    # 3. ENCODING CATEGORICAL VARIABLES
    # -----------------------------
    # Label Encode the target: 'Disaster Type'
    label_encoder = LabelEncoder()
    df['Disaster Type'] = label_encoder.fit_transform(df['Disaster Type'])
    
    # One-hot encode Country, Disaster Group, and Disaster Subgroup
    df = pd.get_dummies(df, columns=['Country', 'Disaster Group', 'Disaster Subgroup'], drop_first=True)
    
    # -----------------------------
    # 3b. DYNAMIC FEATURE SELECTION BASED ON CORRELATION
    # -----------------------------
    cols_to_drop = correlation_feature_selection(df, target_col='Disaster Type', corr_threshold=corr_threshold)
    print("Columns dropped due to high correlation:", cols_to_drop)
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Save the processed dataframe to CSV
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return df

if __name__ == "__main__":
    # Adjust the file paths if needed.
    load_and_preprocess_data("naturalDisasters.xlsx", "processedNaturalDisasters.csv", corr_threshold=0.9)
