FROM ghcr.io/hovoh/python-nltk:latest

WORKDIR /usr/src/app
COPY src .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 9000
CMD ["python", "server.py"]
