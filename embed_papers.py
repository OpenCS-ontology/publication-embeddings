from rdflib import Graph, Literal, Namespace
from rdflib.namespace import XSD, RDF
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import torch
import tqdm
import re

def create_embedding(text_batch, model, tokenizer, padding=True, truncation=True):
    inputs = tokenizer(
        text_batch,
        padding=padding,
        truncation=truncation,
        return_tensors='pt',
        max_length=512,
        return_token_type_ids=False,
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

def add_embedding_to_graph(g :Graph, embedding):
    datacite = Namespace("http://purl.org/spar/datacite/")
    fabio = Namespace("http://purl.org/spar/fabio/")
    ocs_papers = Namespace("https://w3id.org/ocs/ont/papers/")
    g.bind("datacite", datacite)
    g.bind("fabio", fabio)
    g.bind("ocs_papers", ocs_papers)

    paper = g.value(predicate=RDF.type, object=fabio.ResearchPaper)
    g.add(
        (
            paper,
            ocs_papers.hasWordEmbedding,
            Literal(embedding, datatype=XSD.string),
        )
    )
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

    with torch.no_grad():
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
                        abstract, title = re.sub(r'[\r\n]|&nbsp;', '', abstract), re.sub(r'[\r\n]|&nbsp;', '', title)

                        text_batch = [title + tokenizer.sep_token + abstract]

                        embedding = create_embedding(text_batch, model_papers, tokenizer, padding=False)[0]

                        g_emb = add_embedding_to_graph(g, embedding)

                        with open(os.path.join(dir_path_out, ttl_file), "wb") as file:
                            g_emb = g_emb.serialize(format="turtle")
                            if isinstance(g_emb, str):
                                g_emb = g_emb.encode()
                            file.write(g_emb)


if __name__ == "__main__":
    main()
