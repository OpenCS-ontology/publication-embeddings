FROM python:3.10.9-slim-bullseye

COPY requirements.txt /home/requirements.txt
COPY embed_abstracts.py /home/embed_abstracts.py
RUN apt-get update
RUN apt-get install -y git
RUN pip3 install -r /home/requirements.txt

RUN mkdir /home/input_ttl_files
RUN mkdir /home/output_ttl_files

RUN mkdir /home/output_concepts_json

RUN git clone https://github.com/OpenCS-ontology/OpenCS