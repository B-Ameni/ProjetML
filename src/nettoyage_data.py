import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.movies_df = None
        self.credits_df = None
        
    def load_data(self):
        self.movies_df = pd.read_csv(os.path.join(self.data_path, 'movies.csv'))
        self.credits_df = pd.read_csv(os.path.join(self.data_path, 'credits.csv'))
    
    def clean_movies(self):
        
        # 1. Supprimer les doublons
        initial_shape = self.movies_df.shape[0]
        self.movies_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        
        # 2. Supprimer les colonnes non pertinentes ou à forte valeur manquante
        cols_to_drop = []
        for col in self.movies_df.columns:
            missing_pct = (self.movies_df[col].isnull().sum() / len(self.movies_df)) * 100
            if missing_pct > 50:
                cols_to_drop.append(col)
                logger.info(f"Colonne '{col}' supprimée ({missing_pct:.1f}% manquants)")
        
        self.movies_df.drop(columns=cols_to_drop, inplace=True)
        
        # 3. Remplir les valeurs manquantes
        # Pour les colonnes numériques
        numeric_cols = self.movies_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = self.movies_df[col].isnull().sum()
            if missing_count > 0:
                median_val = self.movies_df[col].median()
                self.movies_df[col] = self.movies_df[col].fillna(median_val)        
        # Pour les colonnes texte importantes
        text_cols = ['original_language', 'status']
        for col in text_cols:
            if col in self.movies_df.columns:
                missing_count = self.movies_df[col].isnull().sum()
                if missing_count > 0:
                    self.movies_df[col] = self.movies_df[col].fillna('Unknown')
        
        # 4. Nettoyer les colonnes avec données JSON
        json_cols = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
        for col in json_cols:
            if col in self.movies_df.columns:
                self.movies_df[col] = self.movies_df[col].apply(self._parse_json_column)
        
        # 5. Convertir les types de données
        if 'release_date' in self.movies_df.columns:
            self.movies_df['release_date'] = pd.to_datetime(
                self.movies_df['release_date'], 
                errors='coerce'
            )
        
        # 6. Supprimer les lignes avec id manquant (clé primaire)
        initial_rows = len(self.movies_df)
        self.movies_df.dropna(subset=['id'], inplace=True)
     
        return self.movies_df
    
    def clean_credits(self):
        """Nettoie le dataframe des credits"""
        
        # 1. Supprimer les doublons
        initial_shape = self.credits_df.shape[0]
        self.credits_df.drop_duplicates(subset=['movie_id'], keep='first', inplace=True)
        
        # 2. Remplir les valeurs manquantes dans 'title'
        if 'title' in self.credits_df.columns:
            self.credits_df['title'] = self.credits_df['title'].fillna('Unknown')
        
        # 3. Nettoyer les colonnes JSON
        json_cols = ['cast', 'crew']
        for col in json_cols:
            if col in self.credits_df.columns:
                self.credits_df[col] = self.credits_df[col].apply(self._parse_json_column)
        
        # 4. Supprimer les lignes avec movie_id manquant
        initial_rows = len(self.credits_df)
        self.credits_df.dropna(subset=['movie_id'], inplace=True)
        return self.credits_df
    
    @staticmethod
    def _parse_json_column(value):
        if pd.isnull(value):
            return '[]'
        
        if isinstance(value, str):
            try:
                # Essayer de parser le JSON
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    # Extraire les noms ou les ids
                    names = [item.get('name', item.get('character', '')) for item in parsed]
                    return '|'.join(filter(None, names))
                return str(parsed)
            except (json.JSONDecodeError, TypeError):
                return str(value)
        
        return str(value)
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.data_path
        
            # Sauvegarder les fichiers nettoyés
        movies_output = os.path.join(output_path, 'movies_clean.csv')
        credits_output = os.path.join(output_path, 'credits_clean.csv')
            
        self.movies_df.to_csv(movies_output, index=False, encoding='utf-8')
        self.credits_df.to_csv(credits_output, index=False, encoding='utf-8')
            
        
    
    def get_statistics(self):
        
        if self.movies_df is not None:
            logger.info("\nFilms:")
            logger.info(f"  - Nombre total: {len(self.movies_df)}")
            logger.info(f"  - Colonnes: {list(self.movies_df.columns)}")
            logger.info(f"  - Valeurs manquantes:\n{self.movies_df.isnull().sum()}")
        
        if self.credits_df is not None:
            logger.info("\nCredits:")
            logger.info(f"  - Nombre total: {len(self.credits_df)}")
            logger.info(f"  - Colonnes: {list(self.credits_df.columns)}")
            logger.info(f"  - Valeurs manquantes:\n{self.credits_df.isnull().sum()}")
        
        logger.info("="*50 + "\n")
    
    def run(self, output_path=None):
        try:
            self.load_data()
            self.clean_movies()
            self.clean_credits()
            self.save_cleaned_data(output_path)
            self.get_statistics()
            logger.info("✓ Nettoyage des données terminé avec succès!")
        except Exception as e:
            logger.error(f"✗ Erreur durant le nettoyage: {e}")
            raise


def main():
    """Fonction principale"""
    # Obtenir le chemin du dossier data
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    
    # Créer et exécuter le nettoyeur
    cleaner = DataCleaner(str(data_path))
    cleaner.run()


if __name__ == "__main__":
    main()
