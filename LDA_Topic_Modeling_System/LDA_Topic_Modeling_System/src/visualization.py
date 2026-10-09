"""Plotly and WordCloud visualizations."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


def _dark_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def topic_word_bar(topic_words, topic_number):
    df = pd.DataFrame(topic_words, columns=["Word", "Probability"]).sort_values("Probability")
    fig = px.bar(df, x="Probability", y="Word", orientation="h",
                 title=f"Topic {topic_number} — P(word | topic)")
    return _dark_layout(fig)


def document_topic_bar(probabilities, topic_numbers):
    df = pd.DataFrame({"Topic": topic_numbers, "Probability": probabilities})
    fig = px.bar(df, x="Topic", y="Probability", title="P(topic | document)")
    fig.update_yaxes(range=[0, 1])
    return _dark_layout(fig)


def prevalence_chart(prevalence):
    df = pd.DataFrame({
        "Topic": [f"Topic {i+1}" for i in range(len(prevalence))],
        "Prevalence": prevalence * 100,
    })
    fig = px.bar(df, x="Topic", y="Prevalence", title="Corpus Topic Prevalence (%)")
    return _dark_layout(fig)


def topic_similarity_heatmap(similarity):
    labels = [f"Topic {i+1}" for i in range(len(similarity))]
    fig = px.imshow(similarity, x=labels, y=labels, zmin=0, zmax=1,
                    title="Topic Similarity (cosine)")
    return _dark_layout(fig)


def topic_document_heatmap(doc_topic, max_docs=60):
    matrix = doc_topic[:max_docs]
    fig = px.imshow(
        matrix.T,
        x=[f"DOC-{i+1:04d}" for i in range(matrix.shape[0])],
        y=[f"Topic {i+1}" for i in range(matrix.shape[1])],
        aspect="auto",
        zmin=0,
        zmax=1,
        title=f"Topic–Document Matrix (first {matrix.shape[0]} documents)",
    )
    return _dark_layout(fig)


def wordcloud_figure(topic_words):
    frequencies = {word: float(prob) for word, prob in topic_words}
    wc = WordCloud(width=1200, height=650, background_color="white").generate_from_frequencies(frequencies)
    buffer = io.BytesIO()
    wc.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def topic_similarity_from_distribution(topic_word_distribution):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(topic_word_distribution)
