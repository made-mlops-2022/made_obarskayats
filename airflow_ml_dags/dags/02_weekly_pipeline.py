import os
import pathlib
import datetime
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from airflow.models import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator


default_args={
    'owner': 'airflow',
    'start_date': datetime.datetime(2022, 12, 4),
    'email': ['youremail@gmail.com'],
    'email_on_failure': True,
    'retries':1,
    'retries_delay':datetime.timedelta(minutes=5),
}

def _get_preprocessed_data(dataset_dir: str, output_dir: str):
    datafile_name = os.path.join(dataset_dir, "data.csv")
    targetfile_name = os.path.join(dataset_dir, "target.csv")
    X = pd.read_csv(datafile_name)
    y = pd.read_csv(targetfile_name)
    scaler = StandardScaler()
    X_transform = pd.DataFrame(scaler.fit_transform(X))
    X_transform['target'] = y
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = os.path.join(output_dir, "preprocessed_data.csv")
    X_transform.to_csv(output, index=False)

def _get_train_val_split(output_dir: str, test_split_ratio: float):
    train_file_name = os.path.join(output_dir, "preprocessed_data.csv")
    df = pd.read_csv(train_file_name)
    columns = df.columns[:-1]
    target = df.columns[-1]
    X = df[columns]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split_ratio)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)

def _get_model_fit(output_dir: str, model_dir:str):
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv"))
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

def _get_model_validation(output_dir: str, model_dir:str, predict_dir: str):
    model_dir = os.path.join(model_dir, "model.joblib")
    pathlib.Path(predict_dir).mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_dir)
    X_val = pd.read_csv(os.path.join(output_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(output_dir, "y_val.csv"))
    y_pred = model.predict(X_val)
    y_pred = pd.DataFrame(data=y_pred, columns=['y_predict'])
    y_pred.to_csv(os.path.join(predict_dir, "predictions.csv"), index=False)
    error = mse(y_val, y_pred)
    with open(predict_dir+'/mse_error.csv', 'w') as file:
        file.write(str(error))
    print(error)


with DAG(
    dag_id="02_weekly_pipeline",
    default_args=default_args,
    description='This pipeline implements model training process',
    schedule_interval="@weekly"
) as dag:

    wait_for_raw_data = FileSensor(
        task_id="wait_for_raw_data",
        poke_interval=30,
        filepath="/opt/airflow/data/raw/{{ ds }}"+"/data.csv"
    )
    wait_for_raw_target = FileSensor(
        task_id="wait_for_raw_target",
        poke_interval=30,
        filepath="/opt/airflow/data/raw/{{ ds }}"+"/target.csv"
    )

        # task 1
    preprocessing=PythonOperator(
        task_id='preprocessing_data',
        python_callable=_get_preprocessed_data,
        op_kwargs={
        "year": "{{ execution_date.year }}",
        "month": "{{ execution_date.month }}",
        "day": "{{ execution_date.day }}",
        "hour": "{{ execution_date.hour }}",
        "dataset_dir": "/opt/airflow/data/raw/{{ ds }}",
        "output_dir": "/opt/airflow/data/processed/{{ ds }}"
        }
    )

        # task 2
    train_val_splitting=PythonOperator(
        task_id='train_val_split',
        python_callable=_get_train_val_split,
        op_kwargs={
        "year": "{{ execution_date.year }}",
        "month": "{{ execution_date.month }}",
        "day": "{{ execution_date.day }}",
        "hour": "{{ execution_date.hour }}",
        "output_dir": "/opt/airflow/data/processed/{{ ds }}",
        "test_split_ratio": 0.2,
        }
    )

        # task 3
    model_fitting=PythonOperator(
        task_id='fitting_model',
        python_callable=_get_model_fit,
        email_on_failure=True,
        op_kwargs={
        "year": "{{ execution_date.year }}",
        "month": "{{ execution_date.month }}",
        "day": "{{ execution_date.day }}",
        "hour": "{{ execution_date.hour }}",
        "output_dir": "/opt/airflow/data/processed/{{ ds }}",
        "model_dir": "/opt/airflow/data/models/{{ ds }}"
        }
    )
        # task 4
    model_valudation=PythonOperator(
        task_id='model_validation',
        python_callable=_get_model_validation,
        email_on_failure=True,
        op_kwargs={
        "year": "{{ execution_date.year }}",
        "month": "{{ execution_date.month }}",
        "day": "{{ execution_date.day }}",
        "hour": "{{ execution_date.hour }}",
        "output_dir": "/opt/airflow/data/processed/{{ ds }}",
        "model_dir": "/opt/airflow/data/models/{{ ds }}",
        "predict_dir": "/opt/airflow/data/predictions/{{ ds }}"
        }
    )

    [wait_for_raw_data, wait_for_raw_target] >> preprocessing >> train_val_splitting >> model_fitting >> model_valudation
