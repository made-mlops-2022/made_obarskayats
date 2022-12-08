##### Файл docker-compose.yml взят из репозитория с пары
Для корректной работы с переменными, созданными из UI:

	export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
	export GMAIL_USERNAME=username
	export GMAIL_PASSWORD=password

	docker compose up --build

##### Для запуска 3 дага (03_daily_pipeline) в airflow необходимо прописать переменную 

	MODEL_PATH = /opt/airflow/data/models/

##### Базовое тестирование проведено путем запуска дагов как в примере https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html

python daily_dataset_generator.py 
python 02_weekly_pipeline.py
python daily_dataset_generator.py

##### unittest - можно запустить из папки test следующим скриптом:

python testing.py 
