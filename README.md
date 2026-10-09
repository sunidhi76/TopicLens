# TopicLens — LDA-Based Topic Modeling and Topic Interpretation System

> Discover hidden themes across document collections using probabilistic topic modeling.

TopicLens is an interactive Streamlit application that applies Latent Dirichlet Allocation (LDA) to a collection of text documents. It supports text preprocessing, document-term matrix construction, model training, topic discovery, probabilistic interpretation, and visualization.

## Overview

Topic modeling is an unsupervised NLP technique for discovering recurring themes in a collection of documents. TopicLens represents:
- **Topics as word distributions** — high-probability words are more strongly associated with a topic.
- **Documents as topic mixtures** — a document may contain multiple topics instead of belonging permanently to only one.

## Features

- Built-in BBC News corpus and support for user-uploaded datasets.
- Text preprocessing and stopword filtering.
- Document-Term Matrix (DTM) generation and preview, including dimensions and sparsity.
- Configurable LDA training.
- Topic discovery with top words, generated labels, prevalence, and representative documents.
- Document analysis and corpus visualizations.
- Probabilistic View for topic-word and document-topic distributions.
- Sampling/inference educational view.
- Model evaluation and report/output tools provided by the application.

> Generated topic labels are interpretation aids, not ground-truth dataset categories.

## System Workflow

```text
Document Collection
        |
        v
Text Preprocessing
        |
        v
Document-Term Matrix (CountVectorizer)
        |
        v
LDA Model Training
        |
        v
Latent Topic Discovery
        |
        +---------------------------+
        |                           |
        v                           v
Topic-Word Distribution       Document-Topic Distribution
P(word | topic)               P(topic | document)
        |                           |
        +-------------+-------------+
                      |
                      v
       Topic Interpretation & Visualization
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Streamlit | Interactive web interface |
| scikit-learn | LDA model and CountVectorizer |
| Pandas, NumPy | Data processing and numerical operations |
| NLTK / custom preprocessing | Text cleaning and stopword processing |
| Plotly, Matplotlib | Charts and visualizations |
| WordCloud | Topic-word visualization |

Exact package versions are listed in `requirements.txt`.

## Dataset

The project was tested with the **BBC News Full Text** dataset, which contains articles in five broad categories: business, entertainment, politics, sport, and technology.

LDA is unsupervised. Category labels may be retained as metadata but should **not** be used as target labels during LDA training.

Dataset source: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip

### Preparing the BBC dataset

1. Download and extract the dataset from the source above.
2. Place the extracted `bbc` directory under the project's `data/` directory.
3. Check that it contains category folders such as `business`, `entertainment`, `politics`, `sport`, and `tech`.
4. From the project root, run:

```bash
python prepare_bbc_dataset.py
```

The script prepares the corpus file expected by the application. If the extracted directory differs, check the input path described in the script.

**Note:** The dataset may not be included in this repository. Follow the dataset provider's terms when downloading or redistributing it.

## Project Structure

```text
LDA_Topic_Modeling_System/
├── app.py
├── prepare_bbc_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── bbc/                 # Downloaded/extracted dataset
│   └── bbc_news.csv         # Prepared corpus, if generated
├── src/
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── topic_model.py
│   ├── topic_analysis.py
│   └── ...                  # Other supporting modules
├── assets/                  # Optional assets
└── outputs/                 # Optional generated results
```

This is an illustrative structure; keep it aligned with the files that actually exist in your repository. Do not commit `.venv/`, `__pycache__/`, or unnecessary generated files.

## Installation

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

```bash
git clone https://github.com/YOUR_USERNAME/LDA-Topic-Modeling-System.git
cd LDA-Topic-Modeling-System
```

### 2. Create a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version compatible with the dependencies listed in `requirements.txt`.

## Running the Application

From the project root:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501). Keep the terminal running while using the application.

## How to Use TopicLens

1. Launch the application.
2. Load the prepared BBC corpus or upload a supported dataset.
3. Run preprocessing and inspect the before/after examples.
4. Build the document-term matrix and review its dimensions and sparsity.
5. Configure the number of topics and other available LDA parameters.
6. Train the model.
7. Explore topic summaries, top words, and representative documents.
8. Use Probabilistic View to inspect document-topic and topic-word distributions.
9. Review evaluation metrics and export available results.

When a new corpus is uploaded, the application should preprocess it and train a new LDA model for that corpus.

## Understanding the Results

### Document-Term Matrix

The DTM has one row per document and one column per vocabulary term. Each cell stores a word count. A zero means the term does not appear in that document. The matrix is usually sparse because each document contains only a fraction of the corpus vocabulary.

### Topic-Word Distribution

`P(word | topic)` describes how strongly words are associated with a topic. High-probability terms help interpret the topic's latent theme.

### Document-Topic Distribution

`P(topic | document)` describes the relative contribution of each topic to a document. A document can contain a mixture of several topics.

### Topic Labels

Automatic labels are interpretations derived from discovered words. Review the underlying top words and representative documents; labels are not verified categories.

## Important Implementation Note

The project uses scikit-learn's `LatentDirichletAllocation`, which uses **variational inference**. If the Sampling View discusses Gibbs sampling, it is an educational conceptual explanation—not a claim that the application backend executes Gibbs sampling.

## Limitations

- Topic quality depends on preprocessing, vocabulary filtering, topic count, and model parameters.
- Generated labels may not capture every nuance of a topic.
- Topics can overlap, and documents may have non-zero probability across several topics.
- LDA identifies statistical patterns in word co-occurrence; inspect representative documents when interpreting results.
- Evaluation metrics should be considered alongside qualitative inspection.

## References

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html
2. scikit-learn Developers. *LatentDirichletAllocation Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
3. scikit-learn Developers. *CountVectorizer Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
4. Streamlit. *Streamlit Documentation*. https://docs.streamlit.io/
5. University College Dublin. *BBC News Full Text Dataset*. http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip

## Author

**Project:** TopicLens — LDA-Based Topic Modeling and Topic Interpretation System  
**Course:** Advance NLP (AML23702)  
**Lab Exercise:** 4

Add your name and GitHub profile before publishing.
