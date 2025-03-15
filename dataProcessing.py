import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def drop_low_correlation_features(df, target_col, corr_threshold=0.1):
    """
    Drop features that have a low absolute correlation with the target variable.
    
    Parameters:
        df (DataFrame): The input dataframe.
        target_col (str): The name of the target column.
        corr_threshold (float): The minimum absolute correlation to keep a feature.
        
    Returns:
        List of columns to drop.
    """
    # Select only numeric columns for correlation computation
    numeric_df = df.select_dtypes(include=[np.number])
    # Ensure the target column is included
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]
    
    # Compute the absolute correlations with the target
    target_corr = numeric_df.corr()[target_col].abs()
    
    # Identify features with correlation below the threshold (exclude target)
    drop_cols = target_corr[target_corr < corr_threshold].index.tolist()
    if target_col in drop_cols:
        drop_cols.remove(target_col)
    return drop_cols

def load_and_preprocess_data(input_file, output_file, corr_threshold=0.1):
    # =============================================================================
    # 1. DATA LOADING & INITIAL PREPROCESSING
    # =============================================================================
    df = pd.read_excel(input_file)
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
    
    # =============================================================================
    # 2. DATE CONVERSION & FEATURE ENGINEERING
    # =============================================================================
    # Convert Start Date
    start_date_df = df[['Start Year', 'Start Month', 'Start Day']].replace(-1, np.nan)\
                    .rename(columns={'Start Year': 'year', 'Start Month': 'month', 'Start Day': 'day'})
    df['Start Date'] = pd.to_datetime(start_date_df)
    
    # Convert End Date
    end_date_df = df[['End Year', 'End Month', 'End Day']].replace(-1, np.nan)\
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
    
    # One-hot encode Country, Disaster Group, and Disaster Subgroup
    df = pd.get_dummies(df, columns=['Country', 'Disaster Group', 'Disaster Subgroup'], drop_first=True)
    
    # =============================================================================
    # NEW STEP: DROP LOW-CORRELATION FEATURES
    # =============================================================================
    cols_to_drop = drop_low_correlation_features(df, target_col='Disaster Type', corr_threshold=corr_threshold)
    print("Columns dropped due to low correlation with target:", cols_to_drop)
    
    cols_kept = [col for col in df.columns if col not in cols_to_drop]
    print("Columns kept after dropping low correlation features:", cols_kept)
    
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Save the processed dataframe to CSV
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return df

if __name__ == "__main__":
    load_and_preprocess_data("naturalDisasters.xlsx", "processed_naturalDisasters.csv", corr_threshold=0.1)
