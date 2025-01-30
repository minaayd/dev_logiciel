# data_preprocessing.py

import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    """Load train and test datasets"""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def prepare_features(data):
    """Prepare features for model training"""
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    return pd.get_dummies(data[features])

def get_survival_rates(train_data):
    """Calculate survival rates by gender"""
    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    return sum(women)/len(women), sum(men)/len(men)