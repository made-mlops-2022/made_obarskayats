import pandas as pd
import os.path
import hydra
from ml_project.constants import PROJECT_ROOT
from sklearn.utils.validation import check_is_fitted
from ml_project.models.model_fit_predict import train_model, predict_model, write_csv_data
from ml_project.data.make_dataset import read_data, split_train_val_data, extract_target, extract_features


def test_train_model(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    y_train = extract_target(train_df, cfg)
    X_train = extract_features(train_df, cfg)
    model = train_model(X_train, y_train, cfg.train_params)
    check_is_fitted(model)


def test_prediction_model(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    y_train = extract_target(train_df, cfg)
    X_train = extract_features(train_df, cfg)
    y_val = extract_target(val_df, cfg)
    X_val = extract_features(val_df, cfg)
    model = train_model(X_train, y_train, cfg.train_params)
    predicts = predict_model(model, X_val)
    assert y_val.shape[0] == predicts.shape[0]
    assert type(y_val.values[0]) == type(predicts[0])


def test_written_predictions(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    y_train = extract_target(train_df, cfg)
    X_train = extract_features(train_df, cfg)
    y_val = extract_target(val_df, cfg)
    X_val = extract_features(val_df, cfg)
    model = train_model(X_train, y_train, cfg.train_params)
    predicts = predict_model(model, X_val)
    abs_predicts_path = os.path.join(PROJECT_ROOT, cfg.output_prediction)
    write_csv_data(predicts, abs_predicts_path)
    with open(abs_predicts_path, 'r') as file:
        df = pd.read_csv(abs_predicts_path)
        result = df['y']
        assert result.equals(pd.Series(predicts))


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_config")
def take_arguments_for_test(cfg):
    test_train_model(cfg)
    test_prediction_model(cfg)
    test_written_predictions(cfg)


if __name__ == '__main__':
    take_arguments_for_test()
