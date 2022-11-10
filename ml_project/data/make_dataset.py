import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path):
    df = pd.read_csv(path)
    return df


def split_train_val_data(data, params):
    train_data, val_data = train_test_split(data,
                                            test_size=params['val_size'],
                                            random_state=params['random_state'])
    return train_data, val_data


def extract_target(df, cfg):
    target = df[cfg.feature_params.target]
    return target


def extract_features(df, cfg):
    X_train = df[cfg.feature_params.numerical_features]
    X_train.drop(cfg.feature_params.target, inplace=True, axis=1)
    return X_train
