"""Configurable text preprocessing for TopicLens."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

try:
    from nltk.corpus import stopwords as _stopwords
    STOPWORDS = set(_stopwords.words("english"))
    NEWS_STOPWORDS = {
    "said", "say", "says", "telling", "tell", "told",
    "would", "could", "should", "also",
    "one", "two", "three",
    "year", "years",
    "people", "person",
    "time", "times",
    "new", "news",
    "make", "made", "makes",
    "get", "got", "gets",
    "go", "going", "went",
    "come", "came",
    "take", "took", "takes",
    "use", "used", "uses",
    "like", "well",
    "way", "ways",
    "according",
    "including",
    "number",
    "part",
    "place",
    "thing",
    "things"
    }

    STOPWORDS.update(NEWS_STOPWORDS)

except LookupError:
    STOPWORDS = {
        "a","an","and","are","as","at","be","by","for","from","has","he","in",
        "is","it","its","of","on","that","the","to","was","were","will","with",
        "this","these","those","they","their","them","you","your","i","we","our",
    }

try:
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _LEMMATIZER = None


def ensure_nltk_resources() -> None:
    """Try to make NLTK resources available; failures are handled by fallbacks."""
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass


def _simple_lemma(token: str) -> str:
    """Small fallback lemmatizer for offline use."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _lemma(token: str) -> str:
    if _LEMMATIZER is not None:
        try:
            return _LEMMATIZER.lemmatize(token)
        except LookupError:
            pass
    return _simple_lemma(token)


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_word_length: int = 3,
) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"[a-z]+", text)

    processed = []
    for token in tokens:
        if len(token) < min_word_length:
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        if lemmatize:
            token = _lemma(token)
        if len(token) >= min_word_length:
            processed.append(token)

    return " ".join(processed)


def preprocess_corpus(
    texts: Iterable[str],
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_word_length: int = 3,
):
    ensure_nltk_resources()
    before_counts = []
    processed_texts = []
    for text in texts:
        before = re.findall(r"\b[\w']+\b", str(text))
        processed = preprocess_text(
            text,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            min_word_length=min_word_length,
        )
        before_counts.append(len(before))
        processed_texts.append(processed)

    after_counts = [len(x.split()) for x in processed_texts]
    stats = {
        "original_word_count": int(sum(before_counts)),
        "processed_word_count": int(sum(after_counts)),
        "removed_tokens": int(sum(before_counts) - sum(after_counts)),
        "average_document_length": float(sum(after_counts) / max(len(after_counts), 1)),
    }
    return processed_texts, stats


def add_preprocessed_column(df, **kwargs):
    out = df.copy()
    out["preprocessed_text"], stats = preprocess_corpus(out["text"], **kwargs)
    out = out[out["preprocessed_text"].str.strip().ne("")].reset_index(drop=True)
    return out, stats
