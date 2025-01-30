# tests/test_data_preprocessing.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pandas as pd
import numpy as np
from data_preprocessing import prepare_features, get_survival_rates


def test_prepare_features():
    # Create sample data
    data = pd.DataFrame({
        'Pclass': [1, 2],
        'Sex': ['male', 'female'],
        'SibSp': [1, 0],
        'Parch': [0, 1]
    })
    
    # Test feature preparation
    features = prepare_features(data)
    assert 'Sex_female' in features.columns
    assert 'Sex_male' in features.columns
    assert features.shape[1] == 5  # Pclass, SibSp, Parch, Sex_female, Sex_male

def test_get_survival_rates():
    # Create sample data
    data = pd.DataFrame({
        'Sex': ['male', 'female', 'male', 'female'],
        'Survived': [0, 1, 0, 1]
    })
    
    # Test survival rates calculation
    women_rate, men_rate = get_survival_rates(data)
    assert women_rate == 1.0  # All women survived
    assert men_rate == 0.0    # No men survived