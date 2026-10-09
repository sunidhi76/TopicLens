"""Evaluation and topic-count comparison."""
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


def compare_topic_counts(dtm, topic_counts, max_iter=50, learning_method="batch", random_state=42):
    rows = []
    for n in topic_counts:
        if n > dtm.shape[1]:
            continue
        model = LatentDirichletAllocation(
            n_components=n,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state,
            evaluate_every=-1,
        )
        model.fit(dtm)
        rows.append({
            "Topics": n,
            "Perplexity": float(model.perplexity(dtm)),
            "Log Likelihood": float(model.score(dtm)),
        })
    return pd.DataFrame(rows)
