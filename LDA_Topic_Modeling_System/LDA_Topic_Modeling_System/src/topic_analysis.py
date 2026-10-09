"""Topic interpretation helpers based entirely on model output."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .lda_model import representative_documents


def generated_topic_label(
    top_words: list[tuple[str, float]],
    n_words: int = 3
) -> str:

    semantic_groups = {
        "Computing & Software": {
            "computer", "software", "data", "system",
            "network", "internet", "digital", "technology",
            "web", "online", "program", "programming",
            "server", "email", "security"
        },

        "Football & Sports": {
            "game", "player", "football", "match",
            "club", "league", "goal", "team", "sport",
            "win", "winner", "championship", "coach",
            "season", "olympic"
        },

        "Business & Finance": {
            "company", "firm", "share", "shares",
            "deal", "bank", "business", "market",
            "executive", "investment", "investor",
            "financial", "stock", "profit", "trade"
        },

        "Mobile Technology & Gaming": {
            "mobile", "phone", "game", "games",
            "technology", "digital", "device",
            "service", "network", "music", "console"
        },

        "Law & Justice": {
            "law", "court", "case", "judge",
            "police", "trial", "legal", "crime",
            "rights", "justice", "appeal", "prison"
        },

        "Politics & Government": {
            "government", "labour", "party", "election",
            "minister", "political", "politics",
            "parliament", "policy", "vote", "prime",
            "president", "campaign"
        },

        "Film & Entertainment": {
            "film", "movie", "actor", "actress",
            "award", "music", "show", "star",
            "album", "song", "band", "cinema",
            "director", "entertainment"
        },

        "Economy & Markets": {
            "economy", "economic", "growth", "price",
            "rate", "market", "inflation", "bank",
            "financial", "currency", "trade",
            "rise", "fall", "investment"
        }
    }

    topic_words = {
        word.lower(): float(probability)
        for word, probability in top_words
    }

    scores = {}

    for group_name, group_words in semantic_groups.items():
        score = 0.0

        for word, probability in topic_words.items():
            if word in group_words:
                score += probability

        scores[group_name] = score

    best_group = max(scores, key=scores.get)

    # Require meaningful evidence before assigning a semantic label.
    if scores[best_group] > 0:
        return best_group

    # Generic fallback for datasets outside the BBC domain.
    meaningful_words = [
        word for word, _ in top_words
        if word.lower() not in {
            "said", "say", "says",
            "would", "could", "should",
            "also", "year", "years",
            "people", "person", "time",
            "new", "news"
        }
    ]

    selected_words = meaningful_words[:n_words]

    if not selected_words:
        selected_words = [word for word, _ in top_words[:n_words]]

    return " & ".join(
        word.replace("_", " ").title()
        for word in selected_words
    )

def interpret_topic(
    top_words: list[tuple[str, float]]
) -> str:

    label = generated_topic_label(top_words)

    meaningful_words = [
        word for word, _ in top_words
        if word.lower() not in {
            "said", "say", "says",
            "would", "could", "should",
            "also", "year", "years",
            "people", "person", "time",
            "new", "news"
        }
    ]

    selected_words = meaningful_words[:6]

    if not selected_words:
        selected_words = [
            word for word, _ in top_words[:6]
        ]

    words = ", ".join(selected_words)

    return (
        f"The discovered topic is interpreted as "
        f"'{label}'. Its topic-word distribution is dominated by "
        f"terms such as {words}. These recurring terms indicate "
        f"the main semantic theme represented by this latent topic "
        f"across the document collection. The label is automatically "
        f"generated from the discovered word distribution and is "
        f"not a ground-truth dataset label."
    )


def topic_similarity(topic_word_distribution: np.ndarray) -> np.ndarray:
    return cosine_similarity(topic_word_distribution)


def topic_summary(
    model,
    vectorizer,
    documents,
    doc_topic,
    top_n: int = 10,
    representative_n: int = 5,
) -> list[dict]:
    from .lda_model import get_top_words_per_topic
    top_words = get_top_words_per_topic(model, vectorizer, top_n)
    prevalence = doc_topic.mean(axis=0)
    summaries = []
    for idx, words in enumerate(top_words):
        reps = representative_documents(documents, doc_topic, idx, representative_n)
        summaries.append({
            "topic": idx + 1,
            "label": generated_topic_label(words),
            "top_words": words,
            "prevalence": float(prevalence[idx]),
            "dominant_document_count": int((doc_topic.argmax(axis=1) == idx).sum()),
            "interpretation": interpret_topic(words),
            "representatives": reps,
        })
    return summaries


def topic_word_probability_table(topic_words):
    return pd.DataFrame(
        [{"Word": w, "Probability": p} for w, p in topic_words]
    )
