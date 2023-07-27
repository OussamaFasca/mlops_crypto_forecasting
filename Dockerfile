FROM python:3.11-slim

WORKDIR /app

COPY . /app/

RUN pip3 install -r requirements.txt
RUN pylint src/train.py

CMD ["python3","src/train.py"]