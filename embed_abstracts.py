from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
import os
import pandas as pd
import json
import requests
import shutil

hf_token = "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"

model_id = "sentence-transformers/allenai-specter"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def query(texts):
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
    )
    return response.json()


abstracts = []

archives = ["csis", "scpe"]
input_path = "/home/input_ttl_files"
output_path = "/home/output_ttl_files"

shutil.rmtree(output_path)

data = []
for archive in archives:
    root_dir = os.path.join(input_path, archive)
    for dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir)
        if os.path.isdir(dir_path):
            for ttl_file in os.listdir(dir_path):
                g = Graph()
                g.parse(os.path.join(dir_path, ttl_file), format="ttl")
                result = g.query(
                    """
                        PREFIX dc: <http://purl.org/dc/elements/1.1/>
                        PREFIX fabio: <http://purl.org/spar/fabio/>
                        PREFIX datacite: <http://purl.org/spar/datacite/>
                        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
                        PREFIX dcterms: <http://purl.org/dc/terms/>

                        SELECT ?abstract WHERE {
                        ?paper a fabio:ResearchPaper .
                        ?paper datacite:hasDescription ?blankNode .
                        ?blankNode literal:hasLiteralValue ?abstract .
                        ?paper dcterms:title ?title . 
                        }
                    """
                )
                for row in result:
                    abstract = str(row["abstract"].toPython())
                    embedding = str(query(abstract))

                    datacite = Namespace("http://purl.org/spar/datacite/")
                    bn = Namespace("https://w3id.org/ocs/ont/papers/")
                    g.bind("datacite", datacite)
                    g.bind("", bn)
                    blank_node = g.value(
                        predicate=datacite.hasDescriptionType, object=datacite.abstract
                    )
                    g.add(
                        (
                            blank_node,
                            bn.hasWordEmbedding,
                            Literal(embedding, datatype=XSD.string),
                        )
                    )

                    with open(os.path.join(dir_path, ttl_file), "wb") as file:
                        g = g.serialize(format="turtle")
                        if isinstance(g, str):
                            g = g.encode()
                        file.write(g)
                        print(f"File {ttl_file} embedded")
