# model_training.py

from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y, params=None):
    """Train RandomForest model"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 1
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

def save_model(model, path='model.joblib'):
    """Save trained model"""
    joblib.dump(model, path)