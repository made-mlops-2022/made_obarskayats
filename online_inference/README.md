### main.py
PATH_TO_MODEL=model.joblib PATH_TO_DATA=x_validation.csv uvicorn main:app --host 0.0.0.0 --port 8000
### For building docker image
sudo docker build -t tibarska/model:v6 .
##### Get docker image from docker hub
docker pull tibarska/model:v6
### For running docker:
docker run -p 8000:8000 tibarska/model:v6
### docker hub (docker.io/tibarska/model:v3) page:
https://hub.docker.com/r/tibarska/model


### Selfcheck
1)  done (3)
2) done (1)
3) -
4) -
5) done (4)
6) done (2)
7) done (1)
8) done (1)

Summ: (12)