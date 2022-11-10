import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_model(features, target, train_params):
    mt = train_params.model_type
    if train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=train_params.model_params.RandomForestClassifier.n_estimators,
                                       random_state=train_params.model_params.RandomForestClassifier.random_state,
                                       min_samples_leaf=train_params.model_params.RandomForestClassifier.min_samples_leaf,
                                       max_depth=train_params.model_params.RandomForestClassifier.forest_max_depth)
    elif train_params.model_type == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(random_state=train_params.model_params.DecisionTreeClassifier.random_state,
                                       min_samples_leaf=train_params.model_params.DecisionTreeClassifier.min_samples_leaf,
                                       max_depth=train_params.model_params.DecisionTreeClassifier.max_depth)
    model.fit(features, target)
    return model


def predict_model(model, features):
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts, target):
    f1 = round(f1_score(target, predicts), 3)
    accuracy = round(accuracy_score(target, predicts), 3)
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }


def write_csv_data(data, path):
    data = pd.DataFrame(data, columns=['y'])
    data.to_csv(path, index=False)


def write_json_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)
