import datetime
from airflow.models import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator

default_args={
    'owner': 'airflow',
    'start_date': datetime.datetime(2022, 12, 4),
    'email': ['youremail@gmail.com'],
    'email_on_failure': True,
    'retries':1,
    'retries_delay':datetime.timedelta(minutes=5),
}

DATADIR = "/opt/airflow/data/raw/{{ ds }}"
MOUNT_DATA=Mount(source="/home/tatiana/MLOpls/made_obarskayats/airflow_ml_dags/data/",
                            target="/data",
                            type='bind')

with DAG(
    dag_id="daily_docker_dataset",
    default_args=default_args,
    schedule_interval="@daily"
) as dag:
        # task 1
    daily_dataset=DockerOperator(
        default_args=default_args,
        task_id='docker',
        image="airflow-dataset",
        command=f"--dataset_dir {DATADIR}",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_DATA]
    )

    daily_dataset
