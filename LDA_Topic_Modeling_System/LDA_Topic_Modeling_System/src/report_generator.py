"""Plain-text and CSV report generation."""
from __future__ import annotations

import io
import json


OBJECTIVE = (
    "To apply Latent Dirichlet Allocation on a document collection and interpret "
    "the discovered topics using probabilistic and sampling-based methods."
)


def generate_report(state: dict) -> str:
    lines = [
        "TOPICLENS — TOPIC INTERPRETATION REPORT",
        "=" * 70,
        "",
        "1. PROJECT OBJECTIVE",
        OBJECTIVE,
        "",
        "2. DATASET INFORMATION",
        f"Dataset: {state['dataset_name']}",
        f"Documents: {state['dataset_summary']['documents']}",
        f"Total words: {state['dataset_summary']['total_words']}",
        f"Average document length: {state['dataset_summary']['average_length']:.2f}",
        f"Minimum document length: {state['dataset_summary']['minimum_length']}",
        f"Maximum document length: {state['dataset_summary']['maximum_length']}",
        "LDA training features: article text only; category labels are metadata and are not used.",
        "",
        "3. PREPROCESSING",
        json.dumps(state["preprocessing_config"], indent=2),
        f"Original word count: {state['preprocess_stats']['original_word_count']}",
        f"Processed word count: {state['preprocess_stats']['processed_word_count']}",
        f"Removed tokens: {state['preprocess_stats']['removed_tokens']}",
        f"Average processed document length: {state['preprocess_stats']['average_document_length']:.2f}",
        "",
        "4. DOCUMENT-TERM MATRIX",
        f"Documents: {state['dtm_summary']['documents']}",
        f"Vocabulary size: {state['dtm_summary']['vocabulary_size']}",
        f"Matrix dimensions: {state['dtm_summary']['rows']} x {state['dtm_summary']['columns']}",
        f"Sparsity: {state['dtm_summary']['sparsity']:.4%}",
        "",
        "5. LDA CONFIGURATION",
        json.dumps(state["lda_config"], indent=2),
        "",
        "6. MODEL METRICS",
        f"Log likelihood: {state['metrics']['log_likelihood']:.4f}",
        f"Perplexity: {state['metrics']['perplexity']:.4f}",
        "",
        "7. DISCOVERED TOPICS",
    ]
    for item in state["topic_summaries"]:
        lines += [
            "",
            f"Topic {item['topic']} — {item['label']}",
            f"Prevalence: {item['prevalence']:.2%}",
            f"Dominant-document count: {item['dominant_document_count']}",
            "Top words:",
        ]
        lines += [f"  {word}: {prob:.6f}" for word, prob in item["top_words"]]
        lines += [
            f"Interpretation: {item['interpretation']}",
            "Representative documents:",
        ]
        for _, row in item["representatives"].iterrows():
            lines.append(f"  {row['document_id']} — {row['probability']:.4f} — {row['preview']}")
    lines += [
        "",
        "8. DOCUMENT-TOPIC DISTRIBUTION",
        "Each document is represented as a mixture of the discovered topics.",
        "P(topic | document) is a secondary analysis of topics already discovered by LDA.",
        "",
        "9. PROBABILISTIC AND SAMPLING VIEW",
        "scikit-learn LatentDirichletAllocation uses variational inference.",
        "Gibbs sampling is presented in TopicLens as a conceptual alternative inference approach.",
        "",
        "10. CORPUS-LEVEL OBSERVATIONS",
        "Topics are latent themes discovered from word patterns across the complete document collection.",
        "Topic labels are automatically generated from high-probability words and are not ground-truth labels.",
        "",
        "11. FINAL OBSERVATIONS",
        "Interpretability should be assessed alongside perplexity; a lower perplexity score does not necessarily imply more interpretable topics.",
    ]
    return "\n".join(lines)


def report_bytes(state: dict) -> bytes:
    return generate_report(state).encode("utf-8")
