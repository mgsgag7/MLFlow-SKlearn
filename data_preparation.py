from sklearn.model_selection import train_test_split

def prepare_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test