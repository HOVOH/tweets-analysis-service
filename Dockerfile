FROM ghcr.io/hovoh/mlflow-model-serving:latest

WORKDIR /usr/src/app
RUN pip install nltk~=3.6.2
RUN python -m nltk.downloader all
COPY src .
