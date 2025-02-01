import pandas as pd
import os
import logging
from config import train_data_path, test_data_path, output_directory_path 

def setup_logging():
    """
    Configure le logging pour le script.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données d'entraînement et de test depuis les fichiers CSV.

    Args:
        train_path (str): Chemin vers le fichier CSV des données d'entraînement.
        test_path (str): Chemin vers le fichier CSV des données de test.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames contenant les données d'entraînement et de test.
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Données chargées avec succès.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        raise

def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prépare les données pour l'entraînement en sélectionnant des features et en appliquant one-hot encoding.

    Args:
        train_data (pd.DataFrame): Données d'entraînement.
        test_data (pd.DataFrame): Données de test.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
            - X_train : Données d'entraînement prétraitées.
            - y_train : Labels de l'entraînement.
            - X_test : Données de test prétraitées.
    """
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    
    try:
        X_train = pd.get_dummies(train_data[features])
        X_test = pd.get_dummies(test_data[features])
        y_train = train_data["Survived"]
        logging.info("Prétraitement des données terminé avec succès.")
        return X_train, y_train, X_test
    except KeyError as e:
        logging.error(f"Colonnes manquantes dans les données : {e}")
        raise

def calculate_survival_rate(train_data: pd.DataFrame) -> None:
    """
    Calcule et affiche le taux de survie par sexe.

    Args:
        train_data (pd.DataFrame): Données d'entraînement contenant la colonne 'Survived' et 'Sex'.
    """
    try:
        rate_women = train_data.loc[train_data.Sex == 'female', "Survived"].mean()
        rate_men = train_data.loc[train_data.Sex == 'male', "Survived"].mean()
        logging.info(f"Taux de survie - Femmes : {rate_women:.2%}, Hommes : {rate_men:.2%}")
    except Exception as e:
        logging.error(f"Erreur lors du calcul du taux de survie : {e}")
        raise

def save_data(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> None:
    """
    Sauvegarde les données traitées dans des fichiers CSV.

    Args:
        X_train (pd.DataFrame): Données d'entraînement prétraitées.
        y_train (pd.Series): Labels d'entraînement.
        X_test (pd.DataFrame): Données de test prétraitées.
    """
    try:
        os.makedirs(output_directory_path, exist_ok=True)
        X_train.to_csv(os.path.join(output_directory_path, 'preprocessed_train_features.csv'), index=False)
        y_train.to_csv(os.path.join(output_directory_path, 'preprocessed_train_labels.csv'), index=False)
        X_test.to_csv(os.path.join(output_directory_path, 'preprocessed_test_features.csv'), index=False)
        logging.info("Données sauvegardées avec succès dans le dossier Output.")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des fichiers : {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    
    try:
        train_data, test_data = load_data(train_data_path, test_data_path)
        X_train, y_train, X_test = preprocess_data(train_data, test_data)
        calculate_survival_rate(train_data)
        save_data(X_train, y_train, X_test)
        logging.info("Traitement terminé avec succès.")
        print(X_train.head())
    except Exception as e:
        logging.critical(f"Échec du traitement : {e}")
