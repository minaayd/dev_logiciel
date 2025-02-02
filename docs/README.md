# Les objectifs du projet

Notre projet vise à professionnaliser et industrialiser le code source d'une application d'analyse de données basée sur le jeu de données Titanic de Kaggle. Ce jeu contient des informations sur les passagers du Titanic (âge, classe, genre, etc.) et leur survie, permettant de créer des modèles prédictifs. Notre objectif principal est d'optimiser et de standardiser le code existant selon les meilleures pratiques de l'industrie pour garantir sa maintenabilité et sa qualité.

Notre mission principale est double :

* **Améliorer la qualité et la maintenabilité du code source**
* **Respecter nos contraintes de temps, de qualité et de budget**

Pour atteindre ces objectifs, nous suivrons ces étapes clés :

1. **Restructurer le code en modules Python réutilisables**
2. **Implémenter les standards de développement (PEP 8, tests unitaires, documentation)**
3. **Organiser la collaboration via Git et GitHub**
4. **Mettre en place un pipeline CI/CD pour l'automatisation**

En résumé, ce projet vise à professionnaliser notre base de code en appliquant les meilleures pratiques de développement pour assurer sa qualité et sa pérennité.

# Les instructions d'installation et d'utilisation

Voici un guide étape par étape pour installer et utiliser le projet :

## 1. Préparation des données

Commencez par télécharger les deux fichiers CSV essentiels :

* `train.csv` — Contient les données d'entraînement (891 passagers)
* `test.csv` — Contient les données de test (418 passagers)

Placez-les dans le dossier `data/` du projet.

## 2. Configuration du projet

Une fois les fichiers téléchargés, définissez les chemins d'accès dans le fichier `src/config.py`. Cette centralisation facilite la gestion du projet.

## 3. Installation des outils

Créez un environnement virtuel et installez les dépendances nécessaires :

```shell
python -m venv venv
source venv/bin/activate  # Sur Mac/Linux
venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
```

## 4. Architecture du code

Le projet est divisé en trois modules principaux :

* `data_preprocessing.py` — Gère le nettoyage et la préparation des données
* `model_training.py` — S'occupe de l'entraînement du modèle
* `model_evaluation.py` — Analyse les performances du modèle

## 5. Exécution du projet

Après l'installation, lancez le script principal :

```shell
python src/main.py
```

## 6. Contrôle qualité

Le code est maintenu aux standards de qualité grâce à deux outils :

* **Flake8** — Vérifie le style du code
* **Black** — Assure un formatage cohérent

Vérifiez la qualité du code avec les commandes suivantes :

```shell
flake8 src/
black src/ --check
```

## 7. Structure du projet

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
```

# Prise en main du projet par les utilisateurs

Notre équipe est composée de **3 membres** :

* **Mina** - Chef de Projet
* **Paul** - Développeur Python
* **Bastien** - DevOps

## Mina - Chef de Projet

En tant que chef de projet en alternance, Mina est responsable de la traduction des besoins métiers en spécifications techniques. Elle assure la supervision du projet, le respect des délais, de la qualité et des coûts. Ses responsabilités incluent la rédaction du rapport, la gestion de la documentation README et la coordination d'équipe pour maintenir une communication efficace. Elle a également participé au développement des tests unitaires.

## Paul - Développeur Python

Spécialisé en programmation Python dans le cadre de son alternance, Paul dirige le développement du code et en assure la qualité. Il restructure le notebook initial en modules Python distincts selon les meilleures pratiques. Il implémente les tests unitaires avec `pytest` pour valider chaque composant, notamment les fonctions de prétraitement des données et la logique d'entraînement des modèles. Son travail inclut l'application des standards **PEP 8** et l'utilisation de noms explicites pour les fonctions et variables.

## Bastien - DevOps

Bastien gère le **pipeline CI/CD** via GitHub Actions, qui automatise :

* **Le linting (`Flake8`)** pour la détection d'erreurs
* **Le formatage du code (`Black`)** pour la cohérence
* **Les tests unitaires (`pytest`)** pour la fiabilité

Cette automatisation garantit la qualité du code et facilite le déploiement.

## Une approche véritablement collaborative

Bien que chaque membre ait eu un rôle défini, notre approche a été résolument collaborative. Toutes les décisions importantes ont été prises en équipe, chacun apportant son expertise et son point de vue. Tous les membres ont participé à la revue de code, aux tests et aux discussions techniques. Cette synergie nous a permis d'aboutir à une structure de projet solide et un code de qualité qui reflètent la contribution de chacun.