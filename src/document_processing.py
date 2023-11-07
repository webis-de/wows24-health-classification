#!/usr/bin/env python3
from pathlib import Path
import string

import numpy as np
import pandas as pd
from tira.third_party_integrations import get_output_directory, ir_datasets
from nltk import word_tokenize

DATA_DIR = Path(__file__).parent / "data"


def process_document(document, dw):
    tokens = set(token.lower() for token in word_tokenize(document.default_text()))
    tokens = tokens - set(string.punctuation)
    scores = dw.reindex(list(tokens)).fillna(0).agg(["mean", "median"])
    out = {attr: getattr(document, attr) for attr in document._fields}
    out["mean_health_score"] = round(scores.loc["mean", "encyclopedia"], 4)
    out["median_health_score"] = round(scores.loc["median", "encyclopedia"], 4)
    out["mean_medical_score"] = round(scores.loc["mean", "pubmed"], 4)
    out["median_medical_score"] = round(scores.loc["median", "pubmed"], 4)
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


def process_documents(document_iter):
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
    return pd.DataFrame([process_document(doc, dw) for doc in document_iter])


if __name__ == "__main__":
    # In the TIRA sandbox, this is the injected ir_dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    dataset = ir_datasets.load(
        "workshop-on-open-web-search/document-processing-20231027-training"
    )

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory(".")

    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / "documents.jsonl.gz"

    # You can pass as many additional arguments to your prolgram, e.g., via argparse, to modify the behaviour

    # process the documents, store results at expected location.
    processed_documents = process_documents(dataset.docs_iter())
    processed_documents.to_json(output_file, lines=True, orient="records")
