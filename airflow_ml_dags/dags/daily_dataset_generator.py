import os
import pathlib
import datetime
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.datasets import load_boston

default_args={
    'owner': 'airflow',
    'start_date': datetime.datetime(2022, 12, 4),
    'email': ['youremail@gmail.com'],
    'email_on_failure': True,
    'retries':1,
    'retries_delay':datetime.timedelta(minutes=5),
}

def _get_boston_dataset(dataset_dir: str):
    boston = load_boston()
    df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    df['target'] = pd.DataFrame(boston['target'])
    col_list = list(df.columns)
    df_size = df.shape[0]
    new_df = df.copy()
    df_mean = df.mean()
    df_std = df.std()
    for column in col_list[:-1]:
        col_mean = df_mean[column]
        col_std = df_std[column]
        new_values = np.random.normal(col_mean, col_std, df_size)
        new_df[column] = new_values
    X_data = new_df[col_list[:-1]]
    y_data = new_df['target']
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    x_output = os.path.join(dataset_dir, "data.csv")
    y_output = os.path.join(dataset_dir, "target.csv")
    X_data.to_csv(x_output, index=False)
    y_data.to_csv(y_output, index=False)


with DAG(
        dag_id="01_generate_dataset",
        default_args=default_args,
        schedule_interval="@daily"
) as dag:
    get_boston_dataset = PythonOperator(
        task_id='get_boston_dataset',
        python_callable=_get_boston_dataset,
        email_on_failure=True,
        op_kwargs={
            "dataset_dir": "/opt/airflow/data/raw/{{ ds }}",
        }
    )

    get_boston_dataset

