from rdflib import Graph, Literal, RDF, URIRef, Namespace, BNode
from rdflib.namespace import XSD
from sklearn.metrics.pairwise import cosine_similarity
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
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


abstracts = []

archives = ["csis", "scpe"]
input_path = "/home/input_ttl_files"
output_path = "/home/output_ttl_files"

if os.path.exists(output_path):
  shutil.rmtree(output_path)
os.mkdir(output_path)

data = []
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
                result = g.query(
                    """
                        PREFIX dc: <http://purl.org/dc/elements/1.1/>
                        PREFIX fabio: <http://purl.org/spar/fabio/>
                        PREFIX datacite: <http://purl.org/spar/datacite/>
                        PREFIX literal: <http://www.essepuntato.it/2010/06/literalreification/>
                        PREFIX dcterms: <http://purl.org/dc/terms/>
                        PREFIX prism: <http://prismstandard.org/namespaces/basic/2.0/>
                        PREFIX frbr: <http://purl.org/vocab/frbr/core#>

                        SELECT ?abstract ?title ?doi WHERE {
                        ?paper a fabio:ResearchPaper .
                        ?paper datacite:hasDescription ?blankNode .
                        ?blankNode literal:hasLiteralValue ?abstract .
                        ?paper dcterms:title ?title .
                        ?paper frbr:realization ?article .
                        ?article datacite:doi ?doi .
                        }
                    """
                )
                
                for row in result:
                    dictio = dict()
                    dictio['abstract'] = row['abstract'].toPython()
                    dictio['title'] = row['title'].toPython()
                    dictio['doi'] = row['doi'].toPython()
                    dictio['input_path'] = os.path.join(dir_path, ttl_file)
                    dictio['output_path'] = os.path.join(dir_path_out, ttl_file)
                    print("Before dict")
                    print(dictio)
                    print("After dict")
                    data.append(dictio)


df = pd.DataFrame(data)
embeddings = query(list(df["abstract"]))
df['embedding'] = embeddings
titles = df['title']
dois = df['doi']
print(df.columns)

for index, row in df.iterrows():
    similarities = [cosine_similarity([row['embedding']], [embedding])[0][0] for embedding in df.loc[~df.index.isin([index])]['embedding']]
    similarity_tuples = list(zip(titles, dois, similarities))
    similarity_tuples.sort(key=lambda x: x[2], reverse=True)
    most_similar_papers = similarity_tuples
    num_most_similar = 3 
    top_similar_papers = most_similar_papers[:num_most_similar]

    g = Graph()
    g.parse(row['input_path'], format="ttl")

    datacite = Namespace("http://purl.org/spar/datacite/")
    fabio = Namespace("http://purl.org/spar/fabio/")
    dc = Namespace("http://purl.org/dc/terms/")
    bn = Namespace("https://w3id.org/ocs/ont/papers/")

    g.bind("datacite", datacite)
    g.bind("fabio", fabio)
    g.bind("dc", dc)
    g.bind("", bn)

    paper = g.value(predicate=RDF.type, object=fabio.ResearchPaper)
    blank_node = g.value(predicate=datacite.hasDescriptionType, object=datacite.abstract)
    g.add((blank_node, bn.hasWordEmbedding, Literal(row['embedding'], datatype=XSD.string)))

    for title,doi,similarity in top_similar_papers:
        similar_paper = BNode()
        g.add((paper, bn.isSimilarTo, similar_paper))
        g.add((similar_paper, dc.title, Literal(title, datatype=XSD.string)))
        g.add((similar_paper, datacite.doi, Literal(doi, datatype=XSD.string)))
        g.add((similar_paper, bn.similarityScore, Literal(similarity, datatype=XSD.integer)))

    with open(row['output_path'], "wb") as file:
        g = g.serialize(format="turtle")
        if isinstance(g, str):
            g = g.encode()
        file.write(g)
        print(f"Paper {row['title']} embedded")
