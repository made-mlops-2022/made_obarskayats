import logging
import sys
import os.path
import hydra
from constants import PROJECT_ROOT
from ml_project.data.make_dataset import read_data, split_train_val_data, extract_target, extract_features
from ml_project.models.model_fit_predict import train_model, predict_model, evaluate_model, write_json_data, write_csv_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def run_train_pipeline(cfg):
    logger.info(f"Starting of a ml_project")
    mt = cfg.train_params.model_type
    logger.info(f"Chosen model is {mt}")
    abs_input_path = os.path.join(PROJECT_ROOT, cfg.input_data_path)
    data = read_data(abs_input_path)
    logger.info(f"Raw dataset shape is {data.shape}")
    train_df, val_df = split_train_val_data(data, cfg.splitting_params)
    y_train = extract_target(train_df, cfg)
    X_train = extract_features(train_df, cfg)
    y_val = extract_target(val_df, cfg)
    X_val = extract_features(val_df, cfg)
    logger.info(f"Train dataset shape is {X_train.shape}")
    logger.info(f"Validation dataset shape is {X_val.shape}")
    logger.info(f"Starting of train process with model params {cfg.train_params.model_params[mt]}")
    model = train_model(X_train, y_train, cfg.train_params)
    predicts = predict_model(model, X_val)
    abs_predicts_path = os.path.abspath(cfg.output_prediction)
    write_csv_data(predicts, abs_predicts_path)
    logger.info(f"Predictions of model {mt} are written in {cfg.output_prediction}")
    metrics = evaluate_model(predicts, y_val)
    abs_metrics_path = os.path.abspath(cfg.output_metric_path)
    write_json_data(metrics, abs_metrics_path)
    logger.info(f"Metrics file of model {mt} is written in {cfg.output_metric_path}")
    logger.info(f"Metrics of {mt} are {metrics}")
    return metrics


if __name__ == '__main__':
    run_train_pipeline()
