Для выполнения задания по kubernetes использовался Docker Desktop 
В качестве docker-image выбрала размещенную в docker-hub модель из дз 2

Get docker image from docker hub
docker pull tibarska/model:v6

For running docker:
docker run -p 8000:8000 tibarska/model:v6

docker hub (docker.io/tibarska/model:v3) page:
https://hub.docker.com/r/tibarska/model

Команды:

`kubectl cluster-info`

`kubectl apply -f online-inference-pod.yaml`

`kubectl get pods`