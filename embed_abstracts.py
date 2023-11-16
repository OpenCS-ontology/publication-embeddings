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
import tqdm

def create_embedding(text_batch, model, tokenizer):
    inputs = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=768,
        )
    output = model(**inputs)
    return output.last_hidden_state[:, 0, :].tolist()

def extract_abstract_title(g: Graph):
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

    row = result.bindings[0]
    abstract = row["abstract"].toPython()
    title = row["title"].toPython()

    return abstract, title

def add_embedding_to_graph(g, embedding):
    datacite = Namespace("http://purl.org/spar/datacite/")
    bn = Namespace("https://w3id.org/ocs/ont/papers/")

    g.bind("datacite", datacite)
    g.bind("", bn)

    blank_node = g.value(predicate=datacite.hasDescriptionType, object=datacite.abstract)
    g.add
    ((
        blank_node,
        bn.hasWordEmbedding,
        Literal(embedding, datatype=XSD.string),
    ))
    return g


def main():

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
        if not os.path.exists(root_dir_out):
            os.mkdir(root_dir_out)
        for dir in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir)
            dir_path_out = os.path.join(root_dir_out, dir)
            if not os.path.exists(dir_path_out):
                os.mkdir(dir_path_out)
            if os.path.isdir(dir_path):
                print(f"Embedding papers from {dir_path} ...")
                for ttl_file in tqdm.tqdm(os.listdir(dir_path), total=len(os.listdir(dir_path))):
                    g = Graph()
                    g.parse(os.path.join(dir_path, ttl_file), format="ttl")

                    abstract, title = extract_abstract_title(g)

                    text_batch = [title + tokenizer.sep_token + abstract]

                    embedding = create_embedding(text_batch, model_papers, tokenizer)[0]

                    g = add_embedding_to_graph(g, embedding)

                    with open(os.path.join(dir_path_out, ttl_file), "wb") as file:
                        g = g.serialize(format="turtle")
                        if isinstance(g, str):
                            g = g.encode()
                        file.write(g)


    ONTOLOGY_CORE_DIR = path.join(f"/OpenCS", r"ontology/core")
    output_path_concepts = "/home/output_concepts_json"

    print(f"Reading the OpenCS ontology files from: {ONTOLOGY_CORE_DIR}")
    files = get_all_concept_file_paths(ONTOLOGY_CORE_DIR)
    print(f"Parsing the ontology files")

    graphs = get_graphs_from_files(files)

    concepts_dict = get_concepts_pref_labels(graphs)

    test_dict = get_concepts_all_labels(graphs, concepts_dict)

    concept_texts = list(concepts_dict.values())
    concept_keys = list(concepts_dict.keys())
    article_batch_num = 0
    article_batch_dict = {}
    batch_size = 64
    n = 10
    num_article_batches = len(concept_texts) // (batch_size * n)

    with torch.no_grad():
        for article_batch_num in range(0, num_article_batches):
            article_batch_dict = {}
            start = article_batch_num * batch_size * n
            end = (article_batch_num + 1) * batch_size * n
            print(f"Embedding concepts batch {article_batch_num}/{num_article_batches} ...")
            for i in tqdm.tqdm(range(start, end, batch_size)):
                batch_texts = concept_texts[i:i + batch_size]
                batch_keys = concept_keys[i:i + batch_size]

                embeddings_list = create_embedding(batch_texts, model_concepts, tokenizer)

                for j, key in enumerate(batch_keys):
                    article_batch_dict[key] = test_dict[key]
                    article_batch_dict[key]['embedding'] = embeddings_list[j]
            
            json_object = json.dumps(article_batch_dict, indent=4)
            with open(os.path.join(output_path_concepts, f"opencs_concepts_{article_batch_num}.json"), "w") as outfile:
                outfile.write(json_object)

        #last batch
        article_batch_num += 1
        start = (article_batch_num) * batch_size * n
        end = len(concept_texts)
        for i in tqdm.tqdm(range(start, end, batch_size)):
                batch_end = min(i + batch_size, end)
                batch_texts = concept_texts[i:batch_end]
                batch_keys = concept_keys[i:batch_end]

                embeddings_list = create_embedding(batch_texts, model_concepts, tokenizer)

                for j, key in enumerate(batch_keys):
                    article_batch_dict[key] = test_dict[key]
                    article_batch_dict[key]['embedding'] = embeddings_list[j]
            
        json_object = json.dumps(article_batch_dict, indent=4)
        with open(os.path.join(output_path_concepts, f"opencs_concepts_{article_batch_num}.json"), "w") as outfile:
            outfile.write(json_object)



if __name__ == "__main__":
    main()
