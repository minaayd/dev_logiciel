"""
Module de tests unitaires et d'intégration pour le projet Titanic.
Ce module contient des tests pour vérifier le bon fonctionnement de
toutes les composantes du pipeline de machine learning, du prétraitement
des données jusqu'à l'évaluation du modèle.
"""

import pytest
import time
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model



# Fixtures pour les tests
@pytest.fixture
def sample_data():
    """
    Crée un jeu de données de test pour les différentes fonctions.

    Returns:
        tuple: (train_data, test_data)
            - train_data: Données d'entraînement avec 4 passagers
            - test_data: Données de test avec 2 passagers

    Note:
        Les données incluent les colonnes essentielles du dataset Titanic:
        PassengerId, Survived, Pclass, Sex, SibSp, Parch
    """
    train_data = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Sex": ["male", "female", "female", "male"],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
        }
    )

    test_data = pd.DataFrame(
        {
            "PassengerId": [5, 6],
            "Pclass": [3, 2],
            "Sex": ["female", "male"],
            "SibSp": [0, 0],
            "Parch": [0, 0],
        }
    )

    return train_data, test_data


# Tests pour data_preprocessing.py
def test_preprocess_data(sample_data):
    """
    Teste la fonction de prétraitement des données.

    Args:
        sample_data (tuple): Données de test générées par la fixture

    Vérifie:
        - Les dimensions des données prétraitées
        - La présence des colonnes encodées (one-hot encoding)
        - La structure correcte des données de sortie
    """
    train_data, test_data = sample_data
    X, y, X_test = preprocess_data(train_data, test_data)

    # Vérifier les dimensions
    assert X.shape[0] == 4  # Nombre de lignes train
    assert X_test.shape[0] == 2  # Nombre de lignes test
    assert "Sex_female" in X.columns  # Vérifier l'encodage one-hot
    assert "Sex_male" in X.columns


def test_calculate_survival_rate(sample_data):
    """
    Teste le calcul des taux de survie par genre.

    Args:
        sample_data (tuple): Données de test générées par la fixture

    Vérifie:
        - Le calcul correct du taux de survie des femmes
        - La cohérence des résultats avec les données de test
    """
    train_data, _ = sample_data
    women = train_data.loc[train_data.Sex == "female"]["Survived"]
    rate_women = sum(women) / len(women)
    assert rate_women == 1.0


# Tests pour model_training.py
def test_train_model(sample_data):
    """
    Teste l'entraînement du modèle Random Forest.

    Args:
        sample_data (tuple): Données de test générées par la fixture

    Vérifie:
        - La bonne initialisation du modèle
        - La présence des méthodes essentielles (predict, predict_proba)
        - La capacité du modèle à générer des prédictions
    """
    train_data, test_data = sample_data
    X, y, _ = preprocess_data(train_data, test_data)
    model = train_model(X, y)

    # Vérifier que le modèle est bien entraîné
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # Tester une prédiction
    prediction = model.predict(X)
    assert len(prediction) == X.shape[0]


# Tests pour model_evaluation.py
def test_evaluate_model(sample_data):
    """
    Teste l'évaluation du modèle et la génération des prédictions.

    Args:
        sample_data (tuple): Données de test générées par la fixture

    Vérifie:
        - La cohérence des dimensions des prédictions
        - Le type des prédictions (entiers)
        - Les valeurs des prédictions (0 ou 1)
    """
    train_data, test_data = sample_data
    X, y, X_test = preprocess_data(train_data, test_data)
    model = train_model(X, y)

    predictions = evaluate_model(model, X_test)
    assert len(predictions) == X_test.shape[0]
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    assert all(pred in [0, 1] for pred in predictions)


def test_train_model_performance(sample_data):
    """
    Teste la performance de l'entraînement du modèle, en affichant
    le temps d'entraînement.

    Args:
        sample_data (tuple): Données de test générées par la fixture
    """
    train_data, test_data = sample_data
    X, y, _ = preprocess_data(train_data, test_data)

    # Mesurer le temps d'entraînement
    start_time = time.time()
    model = train_model(X, y)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Temps d'entraînement : {training_time:.2f} secondes")


# Test de l'intégration
def test_full_pipeline(sample_data, tmp_path):
    """
    Teste l'intégration complète du pipeline de machine learning.

    Args:
        sample_data (tuple): Données de test générées par la fixture
        tmp_path (Path): Chemin temporaire fourni par pytest

    Vérifie:
        - Le bon fonctionnement de l'ensemble du pipeline
        - La cohérence des prédictions finales
        - L'intégration correcte de toutes les étapes
    """
    train_data, test_data = sample_data

    # Prétraitement
    X, y, X_test = preprocess_data(train_data, test_data)

    # Entraînement
    model = train_model(X, y)

    # Évaluation
    predictions = evaluate_model(model, X_test)

    # Vérifications finales
    assert len(predictions) == test_data.shape[0]
    assert all(pred in [0, 1] for pred in predictions)
