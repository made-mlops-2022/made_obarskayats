ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY boston_dataset.py ./boston_dataset.py
USER root

ENTRYPOINT ["python", "boston_dataset.py"]