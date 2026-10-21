# TopicLens — LDA-Based Topic Modeling and Topic Interpretation System

> **Discover hidden themes across document collections using probabilistic topic modeling.**

## Objective

**To apply Latent Dirichlet Allocation on a document collection and interpret the discovered topics using probabilistic and sampling-based methods.**

## Expected outcomes

1. Preprocess text data.
2. Build a document-term representation.
3. Train an LDA model.
4. Interpret topic-word distributions.

## What the system demonstrates

TopicLens is deliberately **corpus-first**:

```text
DOCUMENT COLLECTION
        ↓
TEXT PREPROCESSING
        ↓
DOCUMENT-TERM MATRIX
        ↓
LDA
        ↓
HIDDEN TOPICS DISCOVERED ACROSS THE CORPUS
        ↓
TOPIC-WORD DISTRIBUTIONS
        ↓
TOPIC INTERPRETATION
        ↓
CORPUS VISUALIZATION
        ↓
OPTIONAL DOCUMENT-LEVEL ANALYSIS
```

A topic is a latent theme represented as a probability distribution over words. A document is a mixture of discovered topics; it is not permanently assigned to one topic.

- `P(word | topic)` describes a topic's word distribution.
- `P(topic | document)` describes how strongly a discovered topic is represented in a document.
- Document-level analysis is therefore secondary to corpus-level topic discovery.

## Dataset

The primary built-in dataset is the **real BBC News dataset**, commonly distributed as approximately 2,225 documents across Business, Entertainment, Politics, Sport, and Technology.

**Important:** the category labels are metadata only. **LDA is trained only on article text. Category labels are not used as model features.**

### Dataset source / attribution

The BBC dataset is associated with the UCD Machine Learning Group dataset collection and is also widely distributed through dataset repositories such as Kaggle. Obtain the original dataset from its legitimate source and review its applicable license/terms before redistribution.

UCD dataset page:
http://mlg.ucd.ie/datasets/bbc.html

### Dataset preparation

The ZIP intentionally does **not** contain synthetic BBC articles or a fake replacement dataset.

If you have the BBC folder layout:

```text
data/bbc/
├── business/
├── entertainment/
├── politics/
├── sport/
└── tech/
```

run:

```powershell
python prepare_bbc_dataset.py
```

This creates:

```text
data/bbc_news.csv
```

The resulting CSV contains `text` and `category` columns. The application excludes `category` from LDA training.

If your BBC CSV already exists, place it at `data/bbc_news.csv`.

## Technology

- Python 3.13
- Streamlit
- pandas
- NumPy
- scikit-learn
- NLTK
- Plotly
- WordCloud
- Matplotlib
- SciPy

The LDA backend specifically uses:

```python
sklearn.decomposition.LatentDirichletAllocation
```

The document-term matrix uses:

```python
sklearn.feature_extraction.text.CountVectorizer
```

## Installation

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```powershell
streamlit run app.py
```

## Application workflow

1. Select the built-in BBC dataset or upload a CSV/TXT/JSON document collection.
2. Configure preprocessing.
3. Inspect preprocessing statistics and before/after examples.
4. Build the document-term matrix.
5. Configure LDA.
6. Click **Train LDA Model**.
7. Explore the discovered topics.
8. Inspect topic-word probabilities.
9. Inspect representative documents.
10. Optionally inspect a document's topic mixture.
11. Review corpus-level visualizations.
12. Compare topic counts using perplexity and log likelihood.
13. Download the generated reports and CSV outputs.

### User dataset support

Supported formats:

- CSV — must contain a text-like column such as `text`, `article`, `content`, `body`, `document`, `description`, or `news`.
- TXT — blank-line-separated documents are preferred; otherwise each non-empty line is treated as a document.
- JSON — a list of strings, or a list of objects with a supported text-like field. Wrapper keys such as `documents`, `data`, `articles`, and `records` are also supported.

Every new upload receives a **new LDA model**. The BBC-trained model is never reused for uploaded data.

## Text preprocessing

The preprocessing module performs:

- Lowercasing
- URL removal
- Punctuation removal
- Number handling
- Tokenization
- Stopword removal
- Minimum word-length filtering
- Lemmatization
- Whitespace normalization

Configurable options:

- Stopword removal
- Lemmatization
- Minimum word length
- Maximum vocabulary size

TopicLens shows before/after text examples and corpus statistics.

NLTK WordNet resources are attempted automatically. A small deterministic fallback lemmatizer keeps the pipeline usable if the NLTK resource is unavailable.

## Document-term matrix

TopicLens uses `CountVectorizer`.

Default settings:

- `min_df = 2`
- `max_df = 0.95`
- `ngram_range = (1, 1)`

Rows represent documents and columns represent terms. Each cell contains the frequency of a term in a document.

## LDA training

Default configuration:

- Topics: 5
- Maximum iterations: 50
- Learning method: Batch
- Random state: 42
- Top words per topic: 10

Allowed topics: 2–15.

Allowed iterations: 10–200.

Changing a parameter does **not** automatically retrain. Training occurs only after clicking **Train LDA Model**.

The trained model is retained in Streamlit session state so navigation does not retrain it.

## Hidden topic discovery

TopicLens never hard-codes topic names or probabilities.

For each discovered topic, it obtains the model's highest-probability words from `lda.components_` after normalizing each topic's component vector:

```text
P(word | topic)
```

The displayed topic label is automatically generated from those high-probability words. It is an interpretation aid, not a ground-truth category.

## Topic-word distribution

For each topic, TopicLens shows:

- Top words
- Actual probabilities
- Interactive Plotly horizontal bar chart
- Probability table
- Word cloud

The probabilities are calculated from the trained model, not from example values.

## Document-topic distribution

TopicLens calls:

```python
lda.transform(DTM)
```

to obtain:

```text
P(topic | document)
```

A selected document is displayed as a mixture over all discovered topics. This is secondary analysis of corpus-level topics.

## Probabilistic interpretation

TopicLens presents real model output first, then compact explanations:

- `P(topic | document)` — probability that a document exhibits a discovered topic.
- `P(word | topic)` — probability of a word under a discovered topic.

## Variational inference

The implementation uses scikit-learn's `LatentDirichletAllocation`, which performs variational inference.

## Gibbs sampling concept

The Sampling View does **not** claim that scikit-learn's implementation uses Gibbs sampling.

It explains Gibbs sampling conceptually:

1. Start with latent topic assignments.
2. Consider word-topic assignments.
3. Iteratively update latent assignments using conditional probabilities.
4. Repeat until the inferred topic structure stabilizes.

This is presented as an alternative inference approach for LDA.

## Model evaluation

TopicLens calculates:

- Log likelihood
- Perplexity

It can optionally compare:

```text
2, 3, 4, 5, 6, 7, 8, 10 topics
```

Perplexity can help compare models, but a lower perplexity score does not necessarily mean that the topics are more interpretable.

## Required deliverables

The application generates:

1. **Preprocessed corpus** — downloadable CSV.
2. **LDA model output** — configuration, metrics and topic distributions.
3. **Topic-word lists** — downloadable CSV with Topic, Rank, Word and Probability.
4. **Topic interpretation report** — downloadable TXT.
5. **Visualizations** — interactive charts and word clouds.

Generated CSVs include:

- `preprocessed_corpus.csv`
- `topic_word_distribution.csv`
- `document_topic_distribution.csv`

The report contains the objective, dataset information, preprocessing, DTM, LDA configuration, metrics, discovered topics, topic-word probabilities, interpretations, representative documents, document-topic distributions, corpus observations, and final observations.

## Project architecture

```text
LDA_Topic_Modeling_System/
├── app.py
├── prepare_bbc_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── bbc_news.csv          # user-prepared real BBC dataset
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── lda_model.py
│   ├── topic_analysis.py
│   ├── visualization.py
│   ├── evaluation.py
│   └── report_generator.py
├── outputs/
│   └── .gitkeep
└── assets/
    └── .gitkeep
```

The complete backend/model implementation is intentionally not placed in `app.py`.

## Viva questions and answers

### What is LDA?

Latent Dirichlet Allocation is a probabilistic topic-modeling technique that discovers latent topics from a collection of documents.

### What is a topic?

A topic is a latent theme represented as a probability distribution over words.

### Does LDA assign one permanent topic to each document?

No. LDA represents a document as a mixture of topics.

### What is `P(word | topic)`?

It is the probability distribution over words for a particular discovered topic.

### What is `P(topic | document)`?

It is the topic mixture for a document after the corpus-level topics have been discovered.

### Why is the BBC category not passed to LDA?

Because this is unsupervised topic modeling. The purpose is to discover hidden structure from article text rather than train against known categories.

### Does scikit-learn LDA use Gibbs sampling?

No. TopicLens uses scikit-learn's variational inference implementation. Gibbs sampling is shown separately as a conceptual alternative.

### Why use CountVectorizer?

LDA operates on a document-term representation. CountVectorizer creates a matrix of term frequencies where rows are documents and columns are terms.

### Why can perplexity and interpretability disagree?

Perplexity measures model fit/predictive uncertainty and is not a direct measure of human topic coherence or interpretability.

### Why use representative documents?

They provide corpus-level evidence for understanding what each discovered topic means.

## Future enhancements

- Human-in-the-loop topic naming
- Topic coherence metrics
- Interactive topic network graphs
- Better multilingual preprocessing
- Persistent model versioning
- Additional corpora
- Optional supervised comparison against held-out metadata
