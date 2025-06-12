FROM apache/airflow:2.9.1

USER root

RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt /requirements.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements.txt
