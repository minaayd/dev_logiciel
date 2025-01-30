# model_evaluation.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test=None):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    if y_test is not None:
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return predictions, accuracy, report
    return predictions

def save_predictions(predictions, test_data, output_path):
    """Save predictions to CSV"""
    output = pd.DataFrame({
        'PassengerId': test_data.PassengerId,
        'Survived': predictions
    })
    output.to_csv(output_path, index=False)