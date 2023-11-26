import subprocess
from rdflib import Graph, Namespace, RDF
import os
from rapidfuzz import fuzz
import rdflib


def get_embedding_as_list(g):
    default = Namespace("https://w3id.org/ocs/ont/papers/")
    final_embedding = None
    for s, p, embedding in g.triples((None, default.hasWordEmbedding, None)):
        final_embedding = embedding
    return [float(val) for val in final_embedding[1:-1].split(",")]


def are_embeddings_equal(g1, g2):
    emb1 = get_embedding_as_list(g1)
    emb2 = get_embedding_as_list(g2)
    for i in range(len(emb1)):
        if format(emb1[i], ".4f") != format(emb2[i], ".4f"):
            print(emb1[i], emb2[i])
            return False
    return True


if __name__ == "__main__":
    test_out = "/container_test/created_ttl"
    test_in_ttl = "/container_test/true_ttl"
    if not os.path.exists(test_out):
        os.mkdir(test_out)
    for ttl_file in os.listdir(test_in_ttl):
        print(ttl_file)
        final_in_path = os.path.join(test_in_ttl, ttl_file)
        final_out_path = os.path.join(test_out)
        final_output_ttl = ttl_file
        mod_ttl_file = "".join([char for char in ttl_file.lower() if char.isalpha()])
        found_file_flag = False
        for out_ttl in os.listdir(final_out_path):
            print(out_ttl)
            mod_out_ttl = "".join([char for char in out_ttl.lower() if char.isalpha()])
            if fuzz.ratio(mod_ttl_file, mod_out_ttl) > 95:
                final_out_path = os.path.join(test_out, out_ttl)
                found_file_flag = True
                break
        if found_file_flag:
            g1 = Graph()
            g2 = Graph()

            g1.parse(final_in_path, format="ttl")
            g2.parse(final_out_path, format="ttl")
            print(
                "Comparing true Turtle file with the one created during current version test"
            )
            assert are_embeddings_equal(g1, g2)
        print(f"Test with {ttl_file} passed!")
