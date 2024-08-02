from sklearn.model_selection import train_test_split

def preprocess_data(df):
    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.2)
    return train_X, val_X, train_y, val_y
