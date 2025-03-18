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
# Added on 18/3: Helper function to compute the mode (most frequent value).
def mode_func(series):
    """
    Returns the most frequent value (mode) of the given Series.
    If the Series is empty or has no mode, returns NaN.
    """
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan


def load_and_preprocess_data(input_file, output_file, corr_threshold=0.1):
    # =============================================================================
    # 1. DATA LOADING & INITIAL PREPROCESSING
    # =============================================================================
    #Upd on 18/3 removed deaths injuries column. 
    df = pd.read_excel(input_file)
    columns_needed = [
        'Disaster Type', 'Country', 'Start Year', 'Start Month', 'Start Day',
        'End Year', 'End Month', 'End Day','Disaster Group', 'Disaster Subgroup'
    ]
    df = df[columns_needed]
    
    # Fill missing values for numeric columns and assign placeholder for missing date parts
    df = df.fillna({
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
    #=========================================================================
    #New update for 18/3
    # Create Year and Month columns from Start Date
    df['Year'] = df['Start Date'].dt.year
    df['Month'] = df['Start Date'].dt.month
    
        # For binary prediction, mark each disaster occurrence as 1
    df['Disaster Occurred'] = 1

    # =============================================================================
    # 3. GROUPING & AGGREGATION Added on 18/3
    # =============================================================================

    # Group by Country, Year, and Month to aggregate occurrences and average duration
    # - 'Disaster Occurred': Sum then convert to binary.
    # - 'Duration': Mean value.
    # - 'Disaster Group' and 'Disaster Subgroup': Take the mode (most frequent).
    df_grouped = df.groupby(['Country', 'Year', 'Month']).agg({
        'Disaster Occurred': 'sum',
        'Duration': 'mean',
        'Disaster Type': mode_func,
        'Disaster Group': mode_func,      # Added on 18/3: Aggregate Disaster Group by mode.
        'Disaster Subgroup': mode_func    # Added on 18/3: Aggregate Disaster Subgroup by mode.
    }).reset_index()
    
    # Convert counts to binary flag (1 if any disaster occurred, else 0)
    df_grouped['Disaster Occurred'] = (df_grouped['Disaster Occurred'] > 0).astype(int)

    # Drop original date columns (they've been processed)
    df.drop(columns=['Start Year', 'Start Month', 'Start Day',
                     'End Year', 'End Month', 'End Day',
                     'Start Date', 'End Date'], inplace=True)
    
    # =============================================================================
    # 3. ENCODING CATEGORICAL VARIABLES
    # =============================================================================
    # # Label Encode the target: 'Disaster Type'
    # label_encoder = LabelEncoder()
    # df['Disaster Type'] = label_encoder.fit_transform(df['Disaster Type'])
    
    # # One-hot encode Country, Disaster Group, and Disaster Subgroup
    # df = pd.get_dummies(df, columns=['Country', 'Disaster Group', 'Disaster Subgroup'], drop_first=True)
        # =============================================================================
    # 4. COMPLETE GRID FOR COUNTRY YEAR MONTH added on 18/3
    # =============================================================================
    # Get all unique countries from the grouped data
    all_countries = df_grouped['Country'].unique()

    # Convert min and max years to integers
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    all_years = range(min_year, max_year + 1)
    all_months = range(1, 13)
    
    complete_grid = pd.MultiIndex.from_product(
        [all_countries, all_years, all_months],
        names=['Country', 'Year', 'Month']
    ).to_frame(index=False)
    
    # Merge the complete grid with the aggregated data
    df_merged = pd.merge(complete_grid, df_grouped, how='left', on=['Country', 'Year', 'Month'])
    df_merged['Disaster Occurred'] = df_merged['Disaster Occurred'].fillna(0).astype(int)
    df_merged['Duration'] = df_merged['Duration'].fillna(0)

    # !!!!! We keep 'Disaster Type', 'Disaster Group', 'Disaster Subgroup' as strings
    # !!!!! for display. If they are NaN, fill with 'None'.
    df_merged['Disaster Type'] = df_merged['Disaster Type'].fillna('None')  #!!!!!
    df_merged['Disaster Group'] = df_merged['Disaster Group'].fillna('None')  #!!!!!
    df_merged['Disaster Subgroup'] = df_merged['Disaster Subgroup'].fillna('None')  #!!!!!
    
    # One-hot encode Country so the model can use it numerically
    df = pd.get_dummies(df_merged, columns=['Country'], drop_first=True)

    # =============================================================================
    # 5. ENCODING CATEGORICAL VARIABLES (Modified on 18/3)
    # =============================================================================
    # # One-hot encode Country, Disaster Group, and Disaster Subgroup so the model can use them numerically.
    # df_processed = pd.get_dummies(df_merged, columns=['Country', 'Disaster Group', 'Disaster Subgroup'], drop_first=True)
    

    # =============================================================================
    # 5. NEW STEP: DROP LOW-CORRELATION FEATURES modified on 18/3
    # =============================================================================
    # cols_to_drop = drop_low_correlation_features(df, target_col='Disaster Type', corr_threshold=corr_threshold)
    # print("Columns dropped due to low correlation with target:", cols_to_drop)
    
    # cols_kept = [col for col in df.columns if col not in cols_to_drop]
    # print("Columns kept after dropping low correlation features:", cols_kept)
    
    # df.drop(columns=cols_to_drop, inplace=True)
    
    # # Save the processed dataframe to CSV
    # df.to_csv(output_file, index=False)
    # print(f"Processed data saved to {output_file}")
    # return df
        # Use the new target 'Disaster Occurred' for dropping low-correlation features
    country_columns = [col for col in df.columns if col.startswith("Country_")]    
    cols_to_drop = drop_low_correlation_features(df, target_col='Disaster Occurred', corr_threshold=corr_threshold)
    print("Columns dropped due to low correlation with target:", cols_to_drop)
    
    cols_kept = [col for col in df.columns if col not in cols_to_drop]
    print("Columns kept after dropping low correlation features:", cols_kept)
    
    cols_to_drop = [col for col in cols_to_drop if col not in country_columns]
    df.drop(columns=cols_to_drop, inplace=True)
    # =============================================================================
    # 6. SAVE THE PROCESSED DATA
    # =============================================================================
        # Save the processed dataframe to CSV
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return df

if __name__ == "__main__":
    load_and_preprocess_data("naturalDisasters.xlsx", "processedNaturalDisasters.csv", corr_threshold=0.1)
