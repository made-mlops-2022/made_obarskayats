ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

WORKDIR .
RUN python -m pip install --upgrade pip
COPY requirements.txt ./requirements.txt

COPY . .
RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==2.6.0

RUN pip install -r requirements.txt


USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages