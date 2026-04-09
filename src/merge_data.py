import pandas as pd

# Load the datasets
movies_df = pd.read_csv('data/movies_clean.csv')
credits_df = pd.read_csv('data/credits_clean.csv')

# Merge on movie_id (id in movies, movie_id in credits)
merged_df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id', how='inner')

# Drop duplicate columns if any (e.g., title_y if title is duplicated)
merged_df = merged_df.drop(columns=['movie_id', 'title_y'], errors='ignore')
merged_df = merged_df.rename(columns={'title_x': 'title'})

# Save the merged dataset
merged_df.to_csv('data/movies_credits_merged.csv', index=False)

print("Fusion terminée. Fichier sauvegardé : data/movies_credits_merged.csv")
print(f"Nombre de lignes : {len(merged_df)}")
print(f"Colonnes : {list(merged_df.columns)}")