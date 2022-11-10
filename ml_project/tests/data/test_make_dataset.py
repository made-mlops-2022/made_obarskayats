import os.path
from ml_project.constants import PROJECT_ROOT
from ml_project.data.make_dataset import read_data, split_train_val_data
import hydra


def test_load_dataset(cfg):
    target = cfg.feature_params.target
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    assert data.shape[0] > 0
    assert data.shape[1] > 0
    assert target in list(data.columns)
    assert list(data.columns) == cfg.feature_params.numerical_features


def test_split_dataset(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train, val = split_train_val_data(data,  cfg.splitting_params)
    assert train.shape[0] > 0
    assert train.shape[1] > 0
    assert data.shape[0] == val.shape[0] + train.shape[0]
    assert list(train.columns) == list(val.columns)
    assert list(train.columns) == cfg.feature_params.numerical_features


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_config")
def take_arguments_for_test(cfg):
    test_load_dataset(cfg)
    test_split_dataset(cfg)


if __name__ == '__main__':
    take_arguments_for_test()