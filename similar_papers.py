from rdflib import Graph, Literal, RDF, URIRef, Namespace, RDFS
from rdflib.namespace import XSD
from sentence_transformers import SentenceTransformer
import os
import requests
import shutil
from annoy import AnnoyIndex

f = 768  # Length of item vector that will be indexed
t = AnnoyIndex(f, 'angular')


archives = ["csis", "scpe"]
input_path = "/home/input_ttl_files"
output_path = "/home/output_ttl_files"

if os.path.exists(output_path):
  shutil.rmtree(output_path)
os.mkdir(output_path)

for archive in archives:
    root_dir = os.path.join(input_path, archive)
    for dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir)
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
                        PREFIX : <https://w3id.org/ocs/ont/papers/>
                        PREFIX datacite: <http://purl.org/spar/datacite/>
                        PREFIX fabio: <http://purl.org/spar/fabio/>
                        PREFIX frbr: <http://purl.org/vocab/frbr/core#>

                        SELECT ?embedding ?doi WHERE {
                        ?paper a fabio:ResearchPaper .
                        ?paper datacite:hasDescription ?blankNode .
                        ?blankNode :hasWordEmbedding ?embedding .
                        ?paper frbr:realization ?article .
                        ?article datacite:doi ?doi .
                        }
                    """
                )
                
                for row in result:
                    embedding = row['embedding'].toPython()
                    doi = row['doi'].toPython()
                    t.add_item(doi, embedding)



t.build(100) # 10 trees

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

                paper = g.value(predicate=RDF.type, object=fabio.ResearchPaper)

                t.get_nns_by_item(48, 5, search_k=-1, include_distances=False)

                for item_id in range(t.get_n_items()):
                    sim_ids = t.get_nns_by_item(item_id, 4, search_k=-1, include_distances=False)[1:]
                    for sim_id in sim_ids:
                        sim_paper = URIRef(bn.simpaper) 
                        g.add((paper, bn.hasSimPaper, sim_paper))









