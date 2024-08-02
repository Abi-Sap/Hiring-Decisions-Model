from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, val_X, val_y):
    val_prediction = model.predict(val_X)
    rd_accuracy = accuracy_score(val_y, val_prediction)
    rd_accuracy_round = np.round(rd_accuracy, 3)
    print(f'Model accuracy: {rd_accuracy}')
    print(confusion_matrix(val_y, val_prediction))
    print(classification_report(val_y, val_prediction))
    return rd_accuracy_round
