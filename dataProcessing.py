import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def drop_low_correlation_features(df, target_col, corr_threshold=0.1):
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

# Function to handle mode calculation for aggregation
def mode_func(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan

# Function to load and preprocess the data
def load_and_preprocess_data(input_file, output_file, corr_threshold=0.1):
    df = pd.read_excel(input_file)

    # Columns needed for analysis
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

    # Create Year and Month columns from Start Date
    df['Year'] = df['Start Date'].dt.year
    df['Month'] = df['Start Date'].dt.month
    
     # For binary prediction, mark each disaster occurrence as 1
    df['Disaster Occurred'] = 1

    # Group by Country, Year, and Month
    df_grouped = df.groupby(['Country', 'Year', 'Month']).agg({
        'Disaster Occurred': 'sum', # Count the number of disasters
        'Duration': 'mean', # Average duration of disasters
        'Disaster Type': mode_func, # Most frequent disaster type
        'Disaster Group': mode_func, # Most frequent disaster group    
        'Disaster Subgroup': mode_func   # Most frequent disaster subgroup
    }).reset_index() # Convert the grouped data back to a DataFrame
    
    # Convert counts to binary flag (1 if any disaster occurred, else 0)
    df_grouped['Disaster Occurred'] = (df_grouped['Disaster Occurred'] > 0).astype(int)

    # Drop original date columns (they've been processed)
    df.drop(columns=['Start Year', 'Start Month', 'Start Day',
                     'End Year', 'End Month', 'End Day',
                     'Start Date', 'End Date'], inplace=True)
    
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

    # Fill missing values for categorical columns
    df_merged['Disaster Type'] = df_merged['Disaster Type'].fillna('None')  
    df_merged['Disaster Group'] = df_merged['Disaster Group'].fillna('None')  
    df_merged['Disaster Subgroup'] = df_merged['Disaster Subgroup'].fillna('None')
    
    # One-hot encode Country so the model can use it numerically
    df = pd.get_dummies(df_merged, columns=['Country'], drop_first=True)

    # Encode categorical columns using LabelEncoder
    country_columns = [col for col in df.columns if col.startswith("Country_")]    
    cols_to_drop = drop_low_correlation_features(df, target_col='Disaster Occurred', corr_threshold=corr_threshold) # Drop features with low correlation
    print("Columns dropped due to low correlation with target:", cols_to_drop) 
    
    # Keep only the columns with high correlation
    cols_kept = [col for col in df.columns if col not in cols_to_drop]
    print("Columns kept after dropping low correlation features:", cols_kept)
    
    # Drop columns that are not needed for the model
    cols_to_drop = [col for col in cols_to_drop if col not in country_columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Save the processed data to a new file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    return df

# Main function to run the data processing
if __name__ == "__main__":
    load_and_preprocess_data("naturalDisasters.xlsx", "processedNaturalDisasters.csv", corr_threshold=0.1)
