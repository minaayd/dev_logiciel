import subprocess
import logging

# Configurer le logger pour une sortie plus professionnelle
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def run_script(script_name):
    """Exécute un script Python et gère les erreurs avec un log détaillé."""
    try:
        logging.info(f"Début de l'exécution du script {script_name}...")
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        
        # Vérifie si l'exécution du script a échoué et log les erreurs
        if result.returncode != 0:
            logging.error(f"Erreur lors de l'exécution de {script_name}:\n{result.stderr}")
        else:
            logging.info(f"Le script {script_name} a été exécuté avec succès.")
    
    except Exception as e:
        # Capture toute exception et log l'erreur
        logging.error(f"Une exception est survenue lors de l'exécution de {script_name}: {e}")

def main():
    """Fonction principale pour orchestrer l'exécution des scripts du pipeline."""
    scripts = [
        "data_preprocessing.py",  # Prétraitement des données
        "model_training.py",      # Entraînement du modèle
        "model_evaluation.py"     # Évaluation du modèle
    ]

    # Exécution des scripts dans l'ordre
    for script in scripts:
        run_script(script)

    logging.info("Pipeline terminé. Le fichier 'submission.csv' a été créé.")

if __name__ == "__main__":
    main()  
