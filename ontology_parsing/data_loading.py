import glob
from os import path
from typing import List
from rdflib import Graph


def get_all_concept_file_paths(
    ontology_core_dir_path: str, files_format="ttl"
) -> List[str]:
    dirs = glob.glob(path.join(ontology_core_dir_path, "*"))
    files = []
    for i in dirs:
        files += glob.glob(i + f"/*.{files_format}")
    return files


def get_graph_from_file(file_path: str, file_format: str = "ttl") -> Graph:
    graph = Graph()
    graph.parse(file_path, file_format)
    return graph


def get_graphs_from_files(file_paths: List[str], files_format: str = "ttl"):
    graphs = []
    for f in file_paths:
        try:
            graphs.append(get_graph_from_file(f, files_format))
        except:
            print(f"Processing file {f} failed")
    return graphs
