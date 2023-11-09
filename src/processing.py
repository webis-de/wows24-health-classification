#!/usr/bin/env python3
import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from nltk import word_tokenize
from tira.third_party_integrations import get_output_directory, ir_datasets

DATA_DIR = Path(__file__).parent / "data"


def process_item(item, dw):
    tokens = set(token.lower() for token in word_tokenize(item.default_text()))
    tokens = tokens - set(string.punctuation)
    scores = dw.reindex(list(tokens)).fillna(0).agg(["mean", "median"])
    out = {
        "mean_health_score": round(scores.loc["mean", "encyclopedia"], 4),
        "median_health_score": round(scores.loc["median", "encyclopedia"], 4),
        "mean_medical_score": round(scores.loc["mean", "pubmed"], 4),
        "median_medical_score": round(scores.loc["median", "pubmed"], 4),
    }
    return out


def term_domain_specificity(cf: pd.DataFrame, log: float = np.e) -> pd.DataFrame:
    medical = cf.drop("wikipedia", axis=1)
    medical_prob = np.exp(np.log(medical + 1) - np.log(medical.sum()))
    open_domain_prob = np.exp(
        np.log(cf["wikipedia"] + 1) - np.log(cf["wikipedia"].sum())
    )
    out = np.log(medical_prob.div(open_domain_prob, axis=0) + 1) / np.log(log)
    return out


def contrastive_weight(
    cf: pd.DataFrame, log: float = np.e, add: float = 1
) -> pd.DataFrame:
    medical = cf.drop("wikipedia", axis=1)
    data_frequency = medical.add(cf["wikipedia"], axis=0)
    term_1 = medical
    term_2 = data_frequency.sum() / data_frequency
    if log:
        term_1 = np.log(term_1 + add) / np.log(log)
        term_2 = np.log(term_2 + add) / np.log(log)
    out = term_1 * term_2
    return out


def discriminative_weight(
    cf: pd.DataFrame,
    contrastive_log: float = np.e,
    contrastive_add: float = 1,
    specificity_log: float = np.e,
) -> pd.DataFrame:
    contrastive = contrastive_weight(cf, contrastive_log, contrastive_add)
    specificity = term_domain_specificity(cf, specificity_log)
    return contrastive * specificity


def process_iter(document_iter):
    cf = pd.read_parquet(
        DATA_DIR / "wikipedia_unigram_cf.parquet", columns=["corpus_frequency"]
    ).rename(columns={"corpus_frequency": "wikipedia"})
    cf = cf.join(
        pd.read_parquet(
            DATA_DIR / "pubmed_unigram_cf.parquet", columns=["corpus_frequency"]
        ).rename(columns={"corpus_frequency": "pubmed"}),
        how="outer",
    )
    cf = cf.join(
        pd.read_parquet(
            DATA_DIR / "encyclopedia_unigram_cf.parquet", columns=["corpus_frequency"]
        ).rename(columns={"corpus_frequency": "encyclopedia"}),
        how="outer",
    )
    cf = cf.fillna(0)
    dw = discriminative_weight(cf)
    return pd.DataFrame([process_item(doc, dw) for doc in document_iter])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--field", type=str, required=True, choices=["query", "document"]
    )
    args = parser.parse_args()

    dataset = ir_datasets.load(
        "workshop-on-open-web-search/document-processing-20231027-training"
    )

    output_dir = get_output_directory(".")

    if args.field == "query":
        output_file = Path(output_dir) / "queries.jsonl"
        iterator = dataset.queries_iter()
    elif args.field == "document":
        output_file = Path(output_dir) / "documents.jsonl.gz"
        iterator = dataset.docs_iter()
    else:
        raise ValueError(f"Unknown field: {args.field}")

    processed = process_iter(iterator)
    processed.to_json(output_file, lines=True, orient="records")
