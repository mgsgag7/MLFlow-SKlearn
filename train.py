from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from data_gathering import get_data
from data_preparation import prepare_data
import mlflow
import datetime
from sklearn.model_selection import GridSearchCV


rfc = RandomForestClassifier()
dt = DecisionTreeClassifier()

param_rfc = {
    'n_estimators': [10,20,30,40],
    'max_features': ['sqrt','log2'],
    'max_depth': [3,6,9,12,15]
}

param_dt = {
    'criterion': ['entropy','gini'],
    'max_features': ['sqrt','log2'],
    'max_depth': [3,6,9,12,15]
}

models = [rfc,dt]
model_names = ['Random Forest Classifier','Decision Tree']
params = [param_rfc,param_dt]
experiment_name = f"Iris_experiment_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
   

def best_fit_models():
    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        for idx in range(len(models)):
            X,Y = get_data()
            X_train, X_test, y_train, y_test = prepare_data(X,Y)
            gs = GridSearchCV(models[idx],params[idx],scoring="accuracy",cv=5,n_jobs=-1)
            gs_fit = gs.fit(X_train,y_train)
            mlflow.log_metrics({
                "Train_Accuracy": gs_fit.best_score_,
                "Test_Accuracy": gs_fit.score(X_test,y_test)
            })
            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=models[idx],
                artifact_path="iris_model",
                registered_model_name=model_names[idx],
            )