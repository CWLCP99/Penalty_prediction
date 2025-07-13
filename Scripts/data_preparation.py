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

# **Clean up column names** to remove any extra spaces
df.columns = df.columns.str.strip()

# List the columns you expect
check_cols = [
    'Decider?', 'decider_flag',
    'Favourite', 'fav_flag',
    'Great GK?', 'greatGK_flag',
    'Ingame-Shootout?', 'ingame_flag'
]
# 1) Print out which of these are actually present
present = [c for c in check_cols if c in df.columns]
missing = [c for c in check_cols if c not in df.columns]
print("Present columns:", present)
print("Missing columns:", missing)

for col in present:
    print(f"\nValue counts for {col!r}:")
    print(df[col].value_counts(dropna=False))

##### RECODE OF VARIABLES #####
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

#Favorite team?
# === Recode “Favorite team?” ===
#df['fav_flag'] = df['Favourite'].map({'yes': 1, 'no': 0})
# 1) Try converting everything to numeric, coercing errors
df['fav_diff'] = pd.to_numeric(df['Favourite'], errors='coerce')

# 2) Inspect the distribution
#print(df['fav_diff'].describe())
# e.g. count, mean, min, max; you should see values centered around 0, ±1, ±2, etc.

# 3) Create a binary “favored” flag
df['fav_flag'] = (df['fav_diff'] > 0).astype(int)

# 4) Check how many we lost to NaN (the jersey colours etc)
n_bad = df['fav_diff'].isna().sum()
#print(f"Dropped {n_bad} rows where 'Favourite' wasn’t a numeric diff")

# Only keep realistic table‐position differences
df.loc[~df['fav_diff'].between(-2, 2), 'fav_diff'] = np.nan

df['fav_flag'] = (df['fav_diff'] > 0).astype(int)
#print("fav_diff after clipping:\n", df['fav_diff'].describe())
#print("fav_flag counts:\n", df['fav_flag'].value_counts(dropna=False))

# Drop any rows with missing fav_diff (i.e. outliers or non-numeric entries)
df = df.dropna(subset=['fav_diff']) ######## done

# === Recode “Great GK?” ===
# Map only exact “yes”/“no” (case-insensitive), everything else → NaN
df['greatGK_flag'] = df['Great GK?'].str.lower().map({'yes': 1, 'no': 0})

# Show how many valid vs. invalid
#print("Great GK? → flag counts:\n", df['greatGK_flag'].value_counts(dropna=False))

# Drop rows without a valid Great GK? flag
n_before = len(df)
df = df.dropna(subset=['greatGK_flag'])
n_after = len(df)
#print(f"Dropped {n_before - n_after} rows without Great GK? data (keeping {n_after} rows)")

# Is it the deciding kick?
df['decider_flag'] = df['Decider?'].map({'yes': 1, 'no': 0})
# Keep only exact “yes”/“no” (case-insensitive).
#print("Decider? raw → flag counts:\n", df[['Decider?','decider_flag']].dropna().groupby('decider_flag').size())

# Drop NaN rows in 'decider_flag'
df = df.dropna(subset=['greatGK_flag'])

# === Recode “In-game penalty vs shootout?” ===
# In‐game penalty vs. shootout?
df['ingame_flag'] = df['Ingame-Shootout?'].map({
    'Ingame': 1,
    'Shootout': 0
})
print(df['ingame_flag'].value_counts(dropna=False))
df = df.dropna(subset=['ingame_flag'])

# === Recode “Home/Away/Neutral?” ===
# Home/Away/Neutral → two dummies
df['loc_home'] = (df['Location (H-A-N)'] == 'H').astype(int)
df['loc_away'] = (df['Location (H-A-N)'] == 'A').astype(int)

# Last penalty direction (keep numeric)
df['last_dir'] = pd.to_numeric(df['last penalty direction'], errors='coerce')

# Sanity check 
#print(df[['Date', 'foot', 'foot_R', 'age']].head())
#print(df['Decider?', 'decider_flag', 'fav_flag', 'Favourite', 'Great GK?','greatGK_flag', 'Ingame-Shootout?', 'ingame_flag'])  # Check the counts of 'Decider?' and 'decider_flag'
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
#print(df_long[['Choice','alt','chosen']].head(12))
#print(f"Total rows: {len(df_long)},  Chosen count: {df_long['chosen'].sum()}")

# Drop the 'Choice' column as it's no longer needed and keep only relevant columns for simple model. 
keep = [
  'foot_R','age','alt','Choice','chosen',
  'fav_flag','greatGK_flag','ingame_flag'
]

df_model = df_long[keep].copy()

# Verify no missing values remain
#print("Missing in df_model:\n", df_model.isna().sum())
#print("Shape of df_model:", df_model.shape)
#print(df_model.head())

# Save out for Biogeme
#df_model.to_csv('dataset/penalty_long_format.csv', index=False)