FROM python:3.10-slim-buster
RUN python -m pip install --upgrade pip
COPY requirements.txt ./requirements.txt
COPY model.joblib ./model.joblib
COPY x_validation.csv ./x_validation.csv
COPY main.py ./main.py

RUN pip install -r requirements.txt

WORKDIR .
ENV PATH_TO_MODEL="/model.joblib"
ENV PATH_TO_DATA="/x_validation.csv"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
