import os
import pathlib
import random
import datetime
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


def _get_daily_dataset(year: str, month: str, day: str, dataset_dir: str):
    age = []
    id = random.sample(range(1000 * int(day), 2 * 1000 * int(day)), 200)
    n_customers = 200
    for n in range(n_customers):
        age.append(random.randint(20, 60))
    claim_reason = ["Medical", "Travel", "Phone", "Other"]
    claim_reasons = np.random.choice(claim_reason, n_customers, p=[.55, .15, .15, .15])
    variables = [id, age, claim_reasons]
    df = pd.DataFrame(variables).transpose()
    df.columns = ["customer_id", "age", "claim_reason"]
    df["claim_amount"] = 0
    for i in range(len(df)):
        if df["claim_reason"][i] == "Medical":
            df.loc[i, "claim_amount"] = np.random.randint(1300, 2300)
        elif df["claim_reason"][i] == "Travel":
            df.loc[i, 'claim_amount'] = np.random.randint(300, 900)
        elif df["claim_reason"][i] == "Phone":
            df.loc[i, 'claim_amount'] = np.random.randint(200, 270)
        else:
            df.loc[i, 'claim_amount'] = np.random.randint(1, 100)
    dataset_dir = f"{dataset_dir}/{year}_{month}_{day}"
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    output = os.path.join(dataset_dir, "data.csv")
    df.to_csv(output, index=False)


with DAG(
        dag_id="daily_dataset",
        start_date=datetime(2022, 12, 4),
        schedule_interval="@daily"
) as dag:
    get_daily_dataset = PythonOperator(
        task_id='get_df',
        python_callable=_get_daily_dataset,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "hour": "{{ execution_date.hour }}",
            "dataset_dir": "/opt/airflow/data/raw",
        }
    )

    get_daily_dataset
