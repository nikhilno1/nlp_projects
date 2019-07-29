FROM python:3.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 8000

# CMD ["python", "app/server.py", "--task_name", "imdb", "--model_type", "bert", "--model_name_or_path", 
# "https://storage.googleapis.com/deployment-247905.appspot.com/models/imdb_output_final_bert/", "--max_seq_length", "512"]

CMD ["python", "app/server.py", "serve"]