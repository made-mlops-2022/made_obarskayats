import os.path
import joblib
from data.make_dataset import read_data, split_train_val_data, extract_target, extract_features
from models.model_fit_predict import train_model
from constants import PROJECT_ROOT
import os.path
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def train_model_pipeline(cfg):
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    y_train = extract_target(train_df, cfg)
    X_train = extract_features(train_df, cfg)
    y_val = extract_target(val_df, cfg)
    X_val = extract_features(val_df, cfg)
    output_val_path = os.path.join(PROJECT_ROOT, cfg.output_validation)
    X_val.to_csv(output_val_path, index=False)
    model = train_model(X_train, y_train, cfg.train_params)
    abs_output_path = os.path.join(PROJECT_ROOT, cfg.output_model)
    joblib.dump(model, abs_output_path)


if __name__ == '__main__':
    train_model_pipeline()