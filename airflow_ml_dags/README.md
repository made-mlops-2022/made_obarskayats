#### Файл docker-compose.yml взят из репозитория с пары
#### для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build

