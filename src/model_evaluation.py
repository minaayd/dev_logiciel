import os
import joblib
import pandas as pd
import subprocess
import logging
from data_preprocessing import load_data
from config import train_data_path, test_data_path, random_forest_model_path, preprocessed_test_features_path, output_directory_path

# Configuration du logging pour suivre les événements du processus
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Création du dossier de sortie si nécessaire
os.makedirs(output_directory_path, exist_ok=True)

# Vérification si les données prétraitées existent, sinon, exécute le module de formation
if not os.path.exists(preprocessed_test_features_path) and os.path.exists(random_forest_model_path):
    logging.warning("Module 2 non exécuté. Lancement en cours...")
    subprocess.run(["python", "model_training.py"], check=True)

def load_model(filename: str):
    """
    Charge le modèle enregistré à partir du fichier spécifié.

    Args:
        filename (str): Chemin du fichier du modèle sauvegardé.

    Returns:
        object: Le modèle chargé.
    """
    try:
        model = joblib.load(filename)
        logging.info("Modèle chargé avec succès.")
        return model
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame):
    """
    Utilise le modèle pour générer des prédictions sur les données de test.

    Args:
        model (object): Le modèle entraîné.
        X_test (pd.DataFrame): Données de test prétraitées.

    Returns:
        np.ndarray: Les prédictions générées par le modèle.
    """
    try:
        predictions = model.predict(X_test)
        logging.info("Prédictions générées avec succès.")
        return predictions
    except Exception as e:
        logging.error(f"Erreur lors de l'évaluation du modèle : {e}")
        raise

def generate_submission(test_data: pd.DataFrame, predictions):
    """
    Crée et sauvegarde un fichier de soumission avec les identifiants et les prédictions.

    Args:
        test_data (pd.DataFrame): Données de test brutes contenant les identifiants.
        predictions (np.ndarray): Prédictions générées par le modèle.

    Returns:
        None
    """
    try:
        submission_file_path = os.path.join(output_directory_path, 'submission.csv')
        output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
        output.to_csv(submission_file_path, index=False)
        logging.info(f"Soumission sauvegardée sous '{submission_file_path}'.")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de la soumission : {e}")
        raise

def load_preprocessed_data():
    """
    Charge les données de test et leurs caractéristiques prétraitées si elles existent.

    Returns:
        tuple: Données de test brutes et caractéristiques prétraitées.

    Raises:
        FileNotFoundError: Si les fichiers nécessaires ne sont pas trouvés.
    """
    if os.path.exists(preprocessed_test_features_path) and os.path.exists(random_forest_model_path):
        try:
            _, test_data = load_data(train_data_path, test_data_path)  # Chargement des données brutes
            X_test = pd.read_csv(preprocessed_test_features_path)  # Chargement des caractéristiques prétraitées
            logging.info("Données prétraitées chargées avec succès.")
            return test_data, X_test
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données prétraitées : {e}")
            raise
    else:
        logging.error("Problème avec le module 2. Fichiers de données prétraitées manquants.")
        raise FileNotFoundError("Fichiers prétraités non trouvés.")

if __name__ == "__main__":
    try:
        # Chargement des données prétraitées et du modèle
        test_data, X_test = load_preprocessed_data()
        model = load_model(random_forest_model_path)
        
        # Prédictions sur les données de test
        predictions = evaluate_model(model, X_test)
        
        # Création et sauvegarde du fichier de soumission
        generate_submission(test_data, predictions)
        logging.info("Traitement terminé avec succès.")
    except Exception as e:
        logging.critical(f"Échec du traitement : {e}")  
