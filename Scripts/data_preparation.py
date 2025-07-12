# data_preparation.py
import pandas as pd                # data handling
import numpy as np                 # numerical operations
import matplotlib.pyplot as plt # plotting
# Load the dataset
df = pd.read_excel(r'dataset/dataset_raw.xlsx', sheet_name= 'Only relevant 6-Alt-Data', skiprows= 1, header= 0)  # Load the dataset from an Excel file

##### Data Preparation Steps #####
# Show column names
#print('Original column =' + df.columns)

# Parse the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

#print(file_path['Date'].head())# Display the first few entries of the 'Date' column
#print(file_path.head()) 

# Recode foot column values
df['foot_R'] = df['foot'].map({
    'R': 1,  # Right foot
    'L': 0   # Left foot
})
# Display the first few entries of the 'foot_R' column
#print(file_path['foot_R'].head())
# Display the first few rows of the DataFrame
#print(file_path.head())

# Ensure 'Age' column is numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Display the first few entries of the 'age' column
#print(df['age'].head())

# Sanity check 
#print(df[['Date', 'foot', 'foot_R', 'age']].head())
#print("\nMissing values per column:\n", df[['Date','foot_R','age']].isna().sum())

# Approach is to drop rows with missing values in 'Date', 'foot_R', and 'age'
df.dropna(subset=['foot_R', 'age'], inplace=True)
#print("\nMissing values after dropping rows:\n", df[['Date','foot_R','age']].isna().sum())
# Remaining rows after dropping missing values
#print("\nRemaining rows after dropping missing values:", len(df))

#print(df.columns)  # Display the columns of the DataFrame

# Check if 'choice' column exists
#if 'Choice' not in df.columns: 
#    print('Choice column not found')
#else:
#    print('Choice column found')

# Check for missing values in the DataFrame
#for col in df.columns:
    # numpy.isnan only works on numeric; guard with try/except
#    try:
#        n_missing = np.isnan(df[col]).sum()
#    except TypeError:
#        # fall back to pandas for non‐numeric
#        n_missing = df[col].isna().sum()
#    if n_missing > 0:
#        print(f"Column {col!r} has {n_missing} missing values")

############ converting to LONG FORMAT for biogeme 

# Define 6 alternatives
alternatives = [1,2,3,4,5,6] # 6 alternatives for TR, TC, TL, BR, BC, BL

# Build a small DataFrame of alts with a join key
alts_df = pd.DataFrame({
    'alt': alternatives,
    'key': 1
})

# Tag main data with a join key
df['key'] = 1

# Cartesian merge → one row per (observation × alternative)
df_long = df.merge(alts_df, on='key', how='outer').drop(columns='key')

#Flag the chosen zone
#    “Choice” is your 1–6 column
# Chosen sets as 1 where the shooter’s real choice matches that alternative.
df_long['chosen'] = (df_long['Choice'] == df_long['alt']).astype(int)

# Quick sanity checks
print(df_long[['Choice','alt','chosen']].head(12))
print(f"Total rows: {len(df_long)},  Chosen count: {df_long['chosen'].sum()}")

# Drop the 'Choice' column as it's no longer needed and keep only relevant columns for simple model. 
keep = ['foot_R','age','alt','Choice']
df_model = df_long[keep].copy()

# Verify no missing values remain
print("Missing in df_model:\n", df_model.isna().sum())
print("Shape of df_model:", df_model.shape)
print(df_model.head())

# Save out for Biogeme
df_model.to_csv('dataset/penalty_long_format.csv', index=False)