import json
from transformers import AutoTokenizer, AutoModel
from ontology_parsing.graph_utils import (
    get_concepts_pref_labels,
    get_concepts_all_labels,
)
from ontology_parsing.data_loading import (
    get_all_concept_file_paths,
    get_graphs_from_files,
)
from embed_papers import create_embedding
import os
import json
import torch
import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Arguments for testing")
    parser.add_argument("--enable_tests", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    os.environ["CURL_CA_BUNDLE"] = ""

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_aug2023refresh_base")

    model_concepts = AutoModel.from_pretrained("allenai/specter2_aug2023refresh_base")
    model_concepts.load_adapter(
        "allenai/specter2_adhoc_query",
        source="hf",
        load_as="specter2_adhoc_query",
        set_active=True,
    )

    ONTOLOGY_CORE_DIR = os.path.join(f"/OpenCS", r"ontology/core")
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
            if not args.enable_tests:
                article_batch_dict = {}
                start = article_batch_num * batch_size * n
                end = (article_batch_num + 1) * batch_size * n
                print(
                    f"Embedding concepts batch {article_batch_num}/{num_article_batches} ..."
                )
                for i in tqdm.tqdm(range(start, end, batch_size)):
                    batch_texts = concept_texts[i : i + batch_size]
                    batch_keys = concept_keys[i : i + batch_size]

                    embeddings_list = create_embedding(
                        batch_texts, model_concepts, tokenizer
                    )

                    for j, key in enumerate(batch_keys):
                        article_batch_dict[key] = test_dict[key]
                        article_batch_dict[key]["embedding"] = embeddings_list[j]

                json_object = json.dumps(article_batch_dict, indent=4)
                with open(
                    os.path.join(
                        output_path_concepts,
                        f"opencs_concepts_{article_batch_num}.json",
                    ),
                    "w",
                ) as outfile:
                    outfile.write(json_object)

        # last batch
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
                assert len(embeddings_list[j]) == 768
                article_batch_dict[key]["embedding"] = embeddings_list[j]

        json_object = json.dumps(article_batch_dict, indent=4)
        with open(
            os.path.join(
                output_path_concepts, f"opencs_concepts_{article_batch_num}.json"
            ),
            "w",
        ) as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    main()
