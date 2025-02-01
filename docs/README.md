# Projet d'Analyse des Données Titanic

## Objectifs du Projet

Notre projet vise à professionnaliser et industrialiser le code source d'une application d'analyse de données basée sur le jeu de données Titanic de Kaggle<a href="3"/>. 

### Mission Principale
- Améliorer la qualité et la maintenabilité du code source<a href="5"/>
- Respecter nos contraintes de temps, de qualité et de budget<a href="6"/>

### Étapes Clés
1. Restructurer le code en modules Python réutilisables<a href="8"/>
2. Implémenter les standards de développement (PEP 8, tests unitaires, documentation)<a href="9"/>
3. Organiser la collaboration via Git et GitHub<a href="10"/>
4. Mettre en place un pipeline CI/CD pour l'automatisation<a href="11"/>

## Installation et Utilisation

### 1. Préparation des données
- train.csv — Données d'entraînement (891 passagers)<a href="17"/>
- test.csv — Données de test (418 passagers)<a href="18"/>

### 2. Configuration
Définir les chemins d'accès dans `config.py`<a href="20"/>

### 3. Installation des dépendances
```bash
pip install scikit-learn joblib pandas numpy
```<a href="23"/>

### 4. Architecture
Le projet est divisé en trois modules principaux<a href="25"/>:
- `data_preprocessing.py` — Nettoyage et préparation des données<a href="26"/>
- `model_training.py` — Entraînement du modèle<a href="27"/>
- `model_evaluation.py` — Analyse des performances<a href="28"/>

### 5. Contrôle Qualité
Outils utilisés<a href="30"/>:
- Flake8 — Vérification du style<a href="31"/>
- Black — Formatage cohérent<a href="32"/>

### Structure du Projet
```
TITANIC/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── main.py
├── tests/
│   ├── test_all_modules.py
├── docs/
│   ├── README.md
├── data/
│   ├── test.csv
│   └── train.csv
├── requirements.txt
```<a href="34"/>

## Équipe

### Mina - Chef de Projet
Responsable de la gestion de projet, documentation et coordination d'équipe<a href="42"/>

### Paul - Développeur Python
Responsable du développement, implémentation des tests et respect des standards<a href="44"/>

### Bastien - DevOps
Gestion du pipeline CI/CD et automatisation<a href="46"/>:
- Linting (Flake8)<a href="47"/>
- Formatage (Black)<a href="48"/>
- Tests unitaires (pytest)<a href="49"/>