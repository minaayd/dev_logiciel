import subprocess

def run_script(script_name):
    """Fonction pour exécuter un script Python"""
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur dans l'exécution de {script_name}:\n{result.stderr}")
    else:
        print(f"{script_name} exécuté avec succès\n")

if __name__ == "__main__":
    # Exécuter les modules dans l'ordre
    print("Exécution du prétraitement des données...")
    run_script("data_preprocessing.py")
    
    print("Entraînement du modèle...")
    run_script("model_training.py")
    
    print("Évaluation du modèle et création de la soumission...")
    run_script("model_evaluation.py")
    
    print("Pipeline terminé. Le fichier 'submission.csv' a été créé.")

