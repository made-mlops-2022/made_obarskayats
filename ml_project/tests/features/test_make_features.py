import os.path
import hydra
from ml_project.constants import PROJECT_ROOT
from ml_project.data.make_dataset import read_data, split_train_val_data, extract_target, extract_features


def test_extract_target(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    y_train = extract_target(train_df, cfg)
    y_target_train = train_df[cfg.feature_params.target]
    y_val = extract_target(val_df, cfg)
    y_target_val = val_df[cfg.feature_params.target]
    assert y_target_train.equals(y_train)
    assert y_target_val.equals(y_val)


def test_extract_features(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    X_train = extract_features(train_df, cfg)
    X_features_train = train_df[cfg.feature_params.numerical_features]
    X_features_train.drop(cfg.feature_params.target, inplace=True, axis=1)
    X_val = extract_features(val_df, cfg)
    X_features_val = val_df[cfg.feature_params.numerical_features]
    X_features_val.drop(cfg.feature_params.target, inplace=True, axis=1)
    assert X_features_val.equals(X_val)
    assert X_features_train.equals(X_train)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_config")
def take_arguments_for_test(cfg):
    test_extract_target(cfg)
    test_extract_features(cfg)


if __name__ == '__main__':
    take_arguments_for_test()