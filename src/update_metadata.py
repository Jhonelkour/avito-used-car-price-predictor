import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import KFold

# Load the dataset
df = pd.read_csv('avito_final_results.csv')

# Filter out 'autre' brand
df = df[df['Marque'] != 'autre']

# Clean price
df['price'] = (
    df['Price_Raw']
    .astype(str)
    .str.replace(r'\D+', '', regex=True)
    .replace('', np.nan)
    .astype(float)
)
df = df.dropna(subset=['price'])
df['price'] = df['price'].astype(int)

# Calculate brand and model frequencies
brand_frequencies = (df['Marque'].value_counts() / len(df)).to_dict()
model_frequencies = {}
for idx, row in df.groupby(['Marque', 'Modèle']).size().reset_index(name='count').iterrows():
    model_key = f"{row['Marque']}_{row['Modèle']}"
    model_frequencies[model_key] = row['count'] / len(df)

# Calculate target encodings with cross-validation
def target_encoding_cv(df, column, target, n_folds=5):
    """
    Target encoding with cross-validation to prevent overfitting
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    encoded = np.zeros(len(df))
    global_mean = df[target].mean()
    
    for train_idx, val_idx in kf.split(df):
        # Calculate mean target for each category on training fold
        target_mean = df.iloc[train_idx].groupby(column)[target].mean()
        # Apply to validation fold, use global mean for unseen categories
        encoded[val_idx] = df.iloc[val_idx][column].map(target_mean).fillna(global_mean)
    
    return encoded

# Create a copy for encoding
df_temp = df.copy()
df_temp['brand_target_enc'] = target_encoding_cv(df_temp, 'Marque', 'price')
df_temp['model_target_enc_temp'] = target_encoding_cv(df_temp, 'Modèle', 'price')

# Create brand encoding dictionary (average encoding per brand)
brand_encodings = df_temp.groupby('Marque')['brand_target_enc'].mean().to_dict()

# Create model encoding dictionary (average encoding per model)
model_encodings = {}
for marque in df_temp['Marque'].unique():
    brand_df = df_temp[df_temp['Marque'] == marque]
    for modele in brand_df['Modèle'].unique():
        model_key = f"{marque}_{modele}"
        model_encodings[model_key] = brand_df[brand_df['Modèle'] == modele]['model_target_enc_temp'].mean()

# Calculate global mean price
global_mean_price = df['price'].mean()

# Load existing metadata or create new
try:
    with open('model_metadata_improved.pkl', 'rb') as f:
        model_metadata = pickle.load(f)
    print("✓ Loaded existing metadata")
except:
    model_metadata = {}
    print("✓ Creating new metadata dictionary")

# Update metadata with encodings and frequencies
model_metadata['brand_encodings'] = brand_encodings
model_metadata['model_encodings'] = model_encodings
model_metadata['brand_frequencies'] = brand_frequencies
model_metadata['model_frequencies'] = model_frequencies
model_metadata['global_mean_price'] = global_mean_price

# Save updated metadata
with open('model_metadata_improved.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

print("\n" + "=" * 60)
print("METADATA UPDATED SUCCESSFULLY")
print("=" * 60)
print(f"✓ Brand encodings: {len(brand_encodings)} brands")
print(f"✓ Model encodings: {len(model_encodings)} models")
print(f"✓ Brand frequencies: {len(brand_frequencies)} brands")
print(f"✓ Model frequencies: {len(model_frequencies)} models")
print(f"✓ Global mean price: {global_mean_price:,.0f} DH")
print("\nSample brand encodings:")
for brand, enc in list(brand_encodings.items())[:5]:
    print(f"  {brand}: {enc:,.0f} DH")
print("\nSample brand frequencies:")
for brand, freq in list(brand_frequencies.items())[:5]:
    print(f"  {brand}: {freq:.4f}")
print("=" * 60)
