# model_training.py
import os
import joblib
import logging
import pandas as pd
import subprocess
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data
from config import (
    train_data_path,
    test_data_path,
    train_features_path,
    train_labels_path,
    test_features_path,
    output_directory_path,
    rf_model_path,
)

# Configuration du logging pour suivre les événements du processus
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Création du dossier de sortie si nécessaire
os.makedirs(output_directory_path, exist_ok=True)

# Vérification si les données prétraitées existent, sinon exécute le module 1
if not (
    os.path.exists(train_features_path)
    and os.path.exists(train_labels_path)
    and os.path.exists(test_features_path)
):
    logging.warning("Module 1 non exécuté. Lancement en cours...")
    subprocess.run(
        ["python", os.path.join(os.path.dirname(__file__), "data_preprocessing.py")],
        check=True,
    )


def train_model(X, y):
    """Entraîne un modèle RandomForestClassifier.

    Args:
        X (pd.DataFrame): Les caractéristiques d'entraînement.
        y (pd.Series): Les étiquettes correspondantes.

    Returns:
        RandomForestClassifier: Le modèle entraîné.
    """
    try:
        y = y.values.ravel()  # Aplatit les étiquettes en un tableau 1D
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=1
        )  # Initialisation du modèle
        model.fit(X, y)  # Entraînement du modèle
        logging.info("Modèle entraîné avec succès.")
        return model
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement du modèle : {e}")
        raise


def save_model(model, filename):
    """Sauvegarde le modèle entraîné sur le disque.

    Args:
        model (RandomForestClassifier): Le modèle à sauvegarder.
        filename (str): Le chemin du fichier de sauvegarde.
    """
    try:
        joblib.dump(model, filename)  # Sauvegarde du modèle dans un fichier
        logging.info(f"Modèle sauvegardé sous {filename}.")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du modèle : {e}")
        raise


def load_preprocessed_data():
    """Charge les données prétraitées si elles existent.

    Returns:
        tuple: Contient les DataFrames X (caractéristiques d'entraînement),
                                        y (étiquettes),
                                        X_test et test_data.

    Raises:
        FileNotFoundError:
            Si les fichiers de données prétraitées sont introuvables.
    """
    if (
        os.path.exists(train_features_path)
        and os.path.exists(train_labels_path)
        and os.path.exists(test_features_path)
    ):
        try:
            X = pd.read_csv(
                train_features_path
            )  # Chargement des caractéristiques d'entraînement
            y = pd.read_csv(
                train_labels_path
            )  # Chargement des étiquettes d'entraînement
            X_test = pd.read_csv(
                test_features_path
            )  # Chargement des caractéristiques de test
            _, test_data = load_data(
                train_data_path, test_data_path
            )  # Chargement des données brutes de test
            logging.info("Données prétraitées chargées avec succès.")
            return X, y, X_test, test_data  # Retour des données chargées
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données prétraitées : {e}")
            raise
    else:
        logging.error("Problème avec le module 1. Fichiers de prétraitement manquants")
        raise FileNotFoundError(
            "Les fichiers de données prétraitées sont introuvables."
        )


if __name__ == "__main__":
    try:
        # Charger les données prétraitées
        X, y, _, _ = load_preprocessed_data()

        # Entraîner le modèle
        model = train_model(X, y)

        # Sauvegarder le modèle
        save_model(model, rf_model_path)

        logging.info("Modèle entraîné et sauvegardé avec succès.")
    except Exception as e:
        logging.critical(f"Échec du traitement : {e}")
