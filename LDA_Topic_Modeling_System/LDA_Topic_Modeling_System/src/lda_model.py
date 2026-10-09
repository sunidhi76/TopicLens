"""LDA model training and probability extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


def train_lda(
    dtm,
    n_topics: int = 5,
    max_iter: int = 50,
    learning_method: str = "batch",
    random_state: int = 42,
):
    if dtm.shape[0] < 2:
        raise ValueError("LDA requires at least 2 documents.")
    if dtm.shape[1] < n_topics:
        raise ValueError(
            f"The vocabulary has only {dtm.shape[1]} terms; reduce the number of topics."
        )
    model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method=learning_method,
        random_state=random_state,
        evaluate_every=-1,
    )
    model.fit(dtm)
    return model


def get_topic_word_distribution(model) -> np.ndarray:
    components = model.components_.astype(float)
    return components / components.sum(axis=1, keepdims=True)


def get_document_topic_distribution(model, dtm) -> np.ndarray:
    return model.transform(dtm)


def get_top_words_per_topic(model, vectorizer, top_n: int = 10) -> list[list[tuple[str, float]]]:
    dist = get_topic_word_distribution(model)
    vocab = vectorizer.get_feature_names_out()
    results = []
    for topic_idx in range(model.n_components):
        order = np.argsort(dist[topic_idx])[::-1][:top_n]
        results.append([(str(vocab[i]), float(dist[topic_idx, i])) for i in order])
    return results


def topic_word_dataframe(model, vectorizer, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for topic_idx, words in enumerate(get_top_words_per_topic(model, vectorizer, top_n), start=1):
        for rank, (word, prob) in enumerate(words, start=1):
            rows.append({
                "Topic": topic_idx,
                "Rank": rank,
                "Word": word,
                "Probability": prob,
            })
    return pd.DataFrame(rows)


def document_topic_dataframe(model, dtm, documents) -> pd.DataFrame:
    matrix = get_document_topic_distribution(model, dtm)
    df = pd.DataFrame(matrix, columns=[f"Topic {i+1}" for i in range(matrix.shape[1])])
    df.insert(0, "Document ID", documents["document_id"].tolist())
    df["Dominant Topic"] = matrix.argmax(axis=1) + 1
    df["Dominant Probability"] = matrix.max(axis=1)
    return df


def calculate_perplexity(model, dtm) -> float:
    return float(model.perplexity(dtm))


def calculate_log_likelihood(model, dtm) -> float:
    return float(model.score(dtm))


def topic_prevalence(doc_topic: np.ndarray) -> np.ndarray:
    return doc_topic.mean(axis=0)


def representative_documents(documents, doc_topic: np.ndarray, topic_index: int, top_n: int = 5):
    order = np.argsort(doc_topic[:, topic_index])[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append({
            "document_id": documents.iloc[idx]["document_id"],
            "probability": float(doc_topic[idx, topic_index]),
            "preview": str(documents.iloc[idx]["text"]).replace("\n", " ")[:240],
        })
    return pd.DataFrame(rows)
