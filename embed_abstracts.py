from rdflib import Graph, Literal, RDF, Namespace
from rdflib.namespace import XSD
from transformers import AutoTokenizer, AutoModel
from ontology_parsing.graph_utils import get_concepts_pref_labels, get_concepts_all_labels
from ontology_parsing.data_loading import get_all_concept_file_paths, get_graphs_from_files
import os
from os import path
import requests
import shutil
import json
import torch

os.environ["CURL_CA_BUNDLE"] = ""

archives = ["csis", "scpe"]
input_path = "/home/input_ttl_files"
output_path = "/home/output_ttl_files"

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_aug2023refresh_base")

model_papers = AutoModel.from_pretrained("allenai/specter2_aug2023refresh_base")
model_papers.load_adapter(
    "allenai/specter2_aug2023refresh",
    source="hf",
    load_as="specter2_proximity",
    set_active=True,
)

model_concepts = AutoModel.from_pretrained("allenai/specter2_aug2023refresh_base")
model_concepts.load_adapter(
    "allenai/specter2_adhoc_query",
    source="hf",
    load_as="specter2_adhoc_query",
    set_active=True,
)


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
                    abstract = row["abstract"].toPython()
                    title = row["title"].toPython()

                    text_batch = [title + tokenizer.sep_token + abstract]

                    inputs = tokenizer(
                        text_batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        return_token_type_ids=False,
                        max_length=768,
                    )

                    output = model_papers(**inputs)
                    embedding = output.last_hidden_state[:, 0, :].tolist()[0]

                    paper = g.value(predicate=RDF.type, object=fabio.ResearchPaper)
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

                    with open(os.path.join(dir_path_out, ttl_file), "wb") as file:
                        g = g.serialize(format="turtle")
                        if isinstance(g, str):
                            g = g.encode()
                        file.write(g)
                        print(f"Paper '{title}' embedded")



ONTOLOGY_CORE_DIR = path.join(f"/OpenCS", r"ontology/core")
output_path_concepts = "/home/output_concepts_json"

print(f"Reading the OpenCS ontology files from: {ONTOLOGY_CORE_DIR}")
files = get_all_concept_file_paths(ONTOLOGY_CORE_DIR)
print(f"Parsing the ontology files")
# loading the files data into graphs with rdflib
graphs = get_graphs_from_files(files)

# creating a dictionary with concepts and their preferred labels
concepts_dict = get_concepts_pref_labels(graphs)

test_dict = get_concepts_all_labels(graphs, concepts_dict)

counter = 0
batch_num = 0
concept_texts = list(concepts_dict.values())
concept_keys = list(concepts_dict.keys())
batch_size = 64
batch_dict = {}

with torch.no_grad():
    for i in range(0, len(concept_texts), batch_size):
        batch_texts = concept_texts[i:i + batch_size]
        batch_keys = concept_keys[i:i + batch_size]

        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           return_tensors="pt", return_token_type_ids=False, max_length=768)

        output = model_concepts(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]

        embeddings_list = embeddings.tolist()

        for j, key in enumerate(batch_keys):
            batch_dict[key] = test_dict[key]
            batch_dict[key]['embedding'] = embeddings_list[j]

        if counter % 10 == 0:
            batch_num += 1
            json_object = json.dumps(batch_dict, indent=4)
            batch_dict = dict()

            with open(os.path.join(output_path_concepts, f"opencs_concepts_{batch_num}.json"), "w") as outfile:
                outfile.write(json_object)
            
            print(f"Batch {batch_num} embedded")
        
        counter += 1

        if batch_num > 5:
            break
