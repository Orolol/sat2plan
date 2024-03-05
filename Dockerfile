FROM python:3.10.6-buster

WORKDIR /app

COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app
RUN pip install .



