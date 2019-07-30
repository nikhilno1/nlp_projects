FROM python:3.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . app/

EXPOSE 8080

CMD ["python", "app/main.py"]
