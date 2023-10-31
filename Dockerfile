FROM python:3.10.9-slim-bullseye

COPY requirements.txt /home/requirements.txt
COPY embed_abstracts.py /home/embed_abstracts.py

RUN pip3 install -r /home/requirements.txt

RUN mkdir /home/input_ttl_files