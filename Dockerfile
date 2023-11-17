FROM python:3.10.9-slim-bullseye

COPY requirements.txt /home/requirements.txt
COPY embed_papers.py /home/embed_papers.py
COPY embed_concepts.py /home/embed_concepts.py
COPY ontology_parsing /home/ontology_parsing

RUN apt-get update
RUN apt-get install -y git
RUN pip3 install -r /home/requirements.txt

RUN mkdir /home/input_ttl_files
RUN mkdir /home/output_ttl_files
RUN mkdir /home/output_concepts_json

RUN git clone https://github.com/OpenCS-ontology/OpenCS