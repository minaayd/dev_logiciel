import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from model_training import train_model
from model_evaluation import evaluate_model


def test_model_training():
    # Create sample data
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    
    # Test model training
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, 'predict')

def test_model_evaluation():
    # Create sample data and model
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    model = train_model(X, y)
    
    # Test model evaluation
    predictions = evaluate_model(model, X)
    assert len(predictions) == 2
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)