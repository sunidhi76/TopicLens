"""Document-term matrix construction."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def build_document_term_matrix(
    texts,
    min_df: int = 2,
    max_df: float = 0.90,
    max_features: int | None = 5000,
    ngram_range: tuple[int, int] = (1, 1),
):
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    dtm = vectorizer.fit_transform(texts)
    return vectorizer, dtm


def dtm_summary(dtm, vectorizer) -> dict:
    total_cells = dtm.shape[0] * dtm.shape[1]
    sparsity = 1.0 - (dtm.nnz / total_cells) if total_cells else 0.0
    return {
        "documents": int(dtm.shape[0]),
        "vocabulary_size": int(dtm.shape[1]),
        "rows": int(dtm.shape[0]),
        "columns": int(dtm.shape[1]),
        "sparsity": float(sparsity),
    }


def dtm_sample(dtm, vectorizer, n_docs: int = 8, n_terms: int = 12) -> pd.DataFrame:
    terms = vectorizer.get_feature_names_out()[:n_terms]
    arr = dtm[:n_docs, :n_terms].toarray()
    return pd.DataFrame(arr, columns=terms, index=[f"DOC-{i+1:04d}" for i in range(arr.shape[0])])
