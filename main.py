from data_loading import load_data
from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

import warnings
warnings.filterwarnings('ignore')

def main():
    # Load dataset
    df = load_data("recruitment_data.csv")
    
    # Preprocess data
    train_X, val_X, train_y, val_y = preprocess_data(df)
    
    # Train model
    model = train_model(train_X, train_y)
    
    # Evaluate model
    evaluate_model(model, val_X, val_y)

if __name__ == "__main__":
    main()
