import os.path
import joblib
import pandas as pd
import os.path
from constants import PROJECT_ROOT
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def predict_model_pipeline(cfg):
    abs_model_path = os.path.join(PROJECT_ROOT, cfg.output_model)
    model = joblib.load(abs_model_path)
    abs_valid_path = os.path.join(PROJECT_ROOT, cfg.output_validation)
    X_val = pd.read_csv(abs_valid_path)
    predictions = model.predict(X_val)
    abs_pred_path = os.path.join(PROJECT_ROOT, cfg.output_prediction)
    predictions = pd.DataFrame(predictions, columns=['y_pred'])
    predictions.to_csv(abs_pred_path, index=False)


if __name__ == '__main__':
    predict_model_pipeline()

