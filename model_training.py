from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(train_X, train_y):
    model = RandomForestClassifier(random_state=0)
    model.fit(train_X, train_y)
    joblib.dump(model, 'random_model.pkl')
    return model
