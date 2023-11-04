from rdflib import Graph, Literal, RDF, Namespace
from rdflib.namespace import XSD
from sentence_transformers import SentenceTransformer
import os
import requests
import shutil

os.environ['CURL_CA_BUNDLE'] = ''

archives = ["csis", "scpe"]
input_path = "/home/input_ttl_files"
output_path = "/home/output_ttl_files"

if os.path.exists(output_path):
  shutil.rmtree(output_path)
os.mkdir(output_path)

model = SentenceTransformer('sentence-transformers/allenai-specter')

for archive in archives:
    root_dir = os.path.join(input_path, archive)
    root_dir_out = os.path.join(output_path, archive)
    os.mkdir(root_dir_out)
    for dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir)
        dir_path_out = os.path.join(root_dir_out, dir)
        os.mkdir(dir_path_out)
        if os.path.isdir(dir_path):
            for ttl_file in os.listdir(dir_path):
                g = Graph()
                g.parse(os.path.join(dir_path, ttl_file), format="ttl")

                datacite = Namespace("http://purl.org/spar/datacite/")
                fabio = Namespace("http://purl.org/spar/fabio/")
                bn = Namespace("https://w3id.org/ocs/ont/papers/")

                g.bind("datacite", datacite)
                g.bind("fabio", fabio)
                g.bind("", bn)

                result = g.query(
                    """
                        PREFIX fabio: <http://purl.org/spar/fabio/>
                        PREFIX datacite: <http://purl.org/spar/datacite/>
                        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
                        PREFIX dcterms: <http://purl.org/dc/terms/>

                        SELECT ?abstract ?title WHERE {
                        ?paper a fabio:ResearchPaper .
                        ?paper datacite:hasDescription ?blankNode .
                        ?blankNode literal:hasLiteralValue ?abstract .
                        ?paper dcterms:title ?title .
                        }
                    """
                )
                
                for row in result:
                    abstract = row['abstract'].toPython()
                    embedding = model.encode(abstract)

                    paper = g.value(predicate=RDF.type, object=fabio.ResearchPaper)
                    blank_node = g.value(predicate=datacite.hasDescriptionType, object=datacite.abstract)
                    g.add((blank_node, bn.hasWordEmbedding, Literal(embedding, datatype=XSD.string)))

                    with open(os.path.join(dir_path_out, ttl_file), "wb") as file:
                        g = g.serialize(format="turtle")
                        if isinstance(g, str):
                            g = g.encode()
                        file.write(g)
                        print(f"Paper '{row['title']}' embedded")

