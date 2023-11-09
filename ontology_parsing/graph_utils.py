from rdflib import Graph
from typing import List, Tuple, Set, Dict
import rdflib



def get_unique_spo_from_graph(g: Graph) -> Tuple[Set[str]]:
    unique_subjects = []
    unique_predicates = []
    unique_objects = []
    for index, (sub, pred, obj) in enumerate(g):
        unique_subjects.append(sub)
        unique_predicates.append(pred)
        unique_objects.append(obj)
    return set(unique_subjects), set(unique_predicates), set(unique_objects)

def get_unique_spo_from_graphs(graphs: List[Graph]) -> Tuple[Set[str]]:
    sub_sets = []
    pred_sets = []
    objs_sets = []
    for g in graphs:
        g_unique_subs, g_unique_preds, g_unique_objs = get_unique_spo_from_graph(g)
        sub_sets.append(g_unique_subs)
        pred_sets.append(g_unique_preds)
        objs_sets.append(g_unique_objs)
    return set.union(*sub_sets), set.union(*pred_sets), set.union(*objs_sets)

def get_concepts_pref_labels(graphs: List[Graph]) -> Dict[str, str]:
    labels_dict = {}
    for g in graphs:
        for index, (sub, pred, obj) in enumerate(g):
            if pred == rdflib.term.URIRef('http://www.w3.org/2004/02/skos/core#prefLabel'):
                    labels_dict.update({sub.split('/')[-1] : str(obj)})
    return labels_dict

def get_concepts_all_labels(graphs: List[Graph], concepts_lbls: Dict[str, str]) -> Dict[str, str]:
    uris = {
            'http://www.w3.org/2004/02/skos/core#related': 'related',
            'http://www.w3.org/2004/02/skos/core#broader': 'broader'
            }
    uris_keys = [rdflib.term.URIRef(x) for x in list(uris.keys())]

    labels_dict = dict(zip(concepts_lbls.keys(), map(lambda x: {'prefLabel' : x}, concepts_lbls.values())))

    for g in graphs:
        for index, (sub, pred, obj) in enumerate(g):
            if pred in uris_keys:
                concept_id = sub.split('/')[-1]
                pred_label = uris[str(pred)]
                obj_label = concepts_lbls[str(obj).removeprefix('https://w3id.org/ocs/ont/')]
                if pred_label in labels_dict[concept_id]:
                    labels_dict[concept_id][pred_label].append(obj_label)
                else:
                    labels_dict[concept_id][pred_label] = [obj_label]
    return labels_dict

def get_uri_to_colname_dict_from_ontology(graphs, uri_name_split_char='#'):
    _, pred_uris, _ = get_unique_spo_from_graphs(graphs)
    return { str(p): p.split(uri_name_split_char)[-1] for p in pred_uris }