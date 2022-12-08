import os
import datetime
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor


default_args={
    'owner': 'airflow',
    'start_date': datetime.datetime(2022, 12, 4),
    'email': ['youremail@gmail.com'],
    'email_on_failure': True,
    'retries':1,
    'retries_delay':datetime.timedelta(minutes=5),
}

def _get_model_prediction(dataset_dir: str, predict_dir: str, model_dir: str):
    datapath = os.path.join(dataset_dir, "data.csv")
    targetpath = os.path.join(dataset_dir, "target.csv")
    X = pd.read_csv(datapath)
    y = pd.read_csv(targetpath)
    scaler = StandardScaler()
    X_transform = pd.DataFrame(scaler.fit_transform(X))
    X_train, X_val, y_train, y_val = train_test_split(X_transform, y, test_size=0.2)
    MODEL_PATH = Variable.get("MODEL_PATH")
    path = MODEL_PATH + model_dir
    model = joblib.load(path)
    y_pred = model.predict(X_val)
    y_pred = pd.DataFrame(data=y_pred, columns=['y_predict'])
    y_pred.to_csv(os.path.join(predict_dir, "predictions.csv"), index=False)
    error = mse(y_val, y_pred)
    with open(predict_dir+'/mse_error.csv', 'w') as file:
        file.write(str(error))


with DAG(
    dag_id="03_daily_pipeline",
    default_args=default_args,
    description='This pipeline implements model training process',
    schedule_interval="@daily"
) as dag:
    wait_for_model = FileSensor(
        task_id="wait_for_model",
        poke_interval=30,
        filepath="/opt/airflow/data/models/{{ ds }}" + "/model.joblib"
    )

        # task 1
    model_train_daily=PythonOperator(
        task_id='pipeline',
        python_callable=_get_model_prediction,
        op_kwargs={
        "dataset_dir": "/opt/airflow/data/raw/{{ ds }}",
        "predict_dir": "/opt/airflow/data/predictions/{{ ds }}",
        "model_dir": "{{ ds }}/model.joblib"
        }
    )
    wait_for_model >> model_train_daily
