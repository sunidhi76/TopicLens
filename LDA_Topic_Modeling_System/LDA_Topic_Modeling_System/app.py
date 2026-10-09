from __future__ import annotations



import json

from pathlib import Path



import numpy as np

import pandas as pd

import streamlit as st



from src.data_loader import DatasetError, dataset_summary, load_bbc_csv, load_uploaded_file

from src.preprocessing import add_preprocessed_column

from src.vectorization import build_document_term_matrix, dtm_sample, dtm_summary

from src.lda_model import (

    train_lda, get_topic_word_distribution, get_document_topic_distribution,

    get_top_words_per_topic, topic_word_dataframe, document_topic_dataframe,

    calculate_perplexity, calculate_log_likelihood, topic_prevalence,

)

from src.topic_analysis import topic_summary, topic_similarity

from src.visualization import (

    topic_word_bar, document_topic_bar, prevalence_chart,

    topic_similarity_heatmap, topic_document_heatmap, wordcloud_figure,

)

from src.evaluation import compare_topic_counts

from src.report_generator import generate_report



st.set_page_config(

    page_title="TopicLens",

    page_icon="◈",

    layout="wide",

    initial_sidebar_state="expanded",

)



st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght\@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: Inter, sans-serif; }

.stApp { background: #0b1020; color: #f5f7fb; }

[data-testid="stSidebar"] { background: #080d19; border-right: 1px solid #20283a; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }

.hero {

    padding: 28px 30px; border: 1px solid #263149; border-radius: 20px;

    background: linear-gradient(135deg, #121a2c 0%, #0d1424 55%, #111a30 100%);

    margin-bottom: 20px;

}

.hero h1 { font-size: 2.45rem; margin: 0 0 8px; letter-spacing: -1px; }

.hero p { color: #aeb9cf; font-size: 1.05rem; margin: 0; }

.card {

    background: #111827; border: 1px solid #263149; border-radius: 16px;

    padding: 18px; height: 100%; box-shadow: 0 8px 30px rgba(0,0,0,.12);

}

.card h3 { margin-top: 0; }

.badge {

    display:inline-block; padding: 5px 10px; border-radius: 999px;

    background:#1a2842; color:#a9c7ff; font-size:.78rem; font-weight:700;

}

.topic-card {

    background: linear-gradient(180deg,#121c31,#0f1728);

    border:1px solid #2b3955; border-radius:18px; padding:20px; margin-bottom:16px;

}

.topic-label { font-size:1.25rem; font-weight:800; }

.word-chip {

    display:inline-block; background:#19253a; border:1px solid #2a3b5a;

    border-radius:999px; padding:6px 10px; margin:4px 4px 0 0; color:#dce6f8;

}

.small { color:#94a3b8; font-size:.88rem; }

.flow {

    display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:18px 0;

}

.flow span {

    padding:10px 13px; border:1px solid #2a3853; border-radius:12px; background:#10192b;

    font-size:.86rem; font-weight:700;

}

.metric-note { color:#94a3b8; font-size:.8rem; margin-top:4px; }

</style>

""", unsafe_allow_html=True)



OBJECTIVE = ("To apply Latent Dirichlet Allocation on a document collection and interpret "

             "the discovered topics using probabilistic and sampling-based methods.")



PAGES = [

    "Overview", "Dataset", "Preprocessing", "Document-Term Matrix",

    "LDA Configuration", "Train Model", "Topic Discovery", "Topic Explorer",

    "Document Analysis", "Corpus Visualization", "Model Evaluation",

    "Probabilistic View", "Sampling View", "Reports"

]





def init_state():

    defaults = {

        "dataset": None, "dataset_name": None, "preprocessed": None,

        "preprocess_stats": None, "preprocessing_config": None,

        "vectorizer": None, "dtm": None, "dtm_summary": None,

        "lda": None, "doc_topic": None, "topic_word": None,

        "topic_summaries": None, "metrics": None, "lda_config": None,

        "last_training_error": None,

    }

    for k, v in defaults.items():

        if k not in st.session_state:

            st.session_state[k] = v





def reset_model_state():

    for k in ["preprocessed", "preprocess_stats", "preprocessing_config",

              "vectorizer", "dtm", "dtm_summary", "lda", "doc_topic",

              "topic_word", "topic_summaries", "metrics", "lda_config",

              "last_training_error"]:

        st.session_state[k] = None





def trained() -> bool:

    return st.session_state.lda is not None





def status_badge():

    if trained():

        st.sidebar.markdown('<span class="badge">MODEL STATUS · LDA trained ✓</span>', unsafe_allow_html=True)

    else:

        st.sidebar.markdown('<span class="badge">MODEL STATUS · Not trained</span>', unsafe_allow_html=True)





def require_training():

    if not trained():

        st.info("Your LDA model hasn't been trained yet. Select a dataset, configure preprocessing and LDA, then click **Train LDA Model**.")

        return False

    return True





def overview():

    st.markdown("""

    <div class="hero">

      <h1>TopicLens</h1>

      <p>Discover hidden themes across document collections using probabilistic topic modeling.</p>

    </div>

    """, unsafe_allow_html=True)

    st.markdown(f"### Academic objective\n> {OBJECTIVE}")

    st.markdown("""

    <div class="flow">

      <span>DOCUMENT COLLECTION</span> → <span>PREPROCESSING</span> → <span>DOCUMENT-TERM MATRIX</span>

      → <span>LDA</span> → <span>HIDDEN TOPICS</span> → <span>TOPIC INTERPRETATION</span>

    </div>

    """, unsafe_allow_html=True)

    cols = st.columns(4)

    outcomes = [

        ("01", "Preprocess text data", bool(st.session_state.preprocessed is not None)),

        ("02", "Build a document-term representation", bool(st.session_state.dtm is not None)),

        ("03", "Train an LDA model", trained()),

        ("04", "Interpret topic-word distributions", trained()),

    ]

    for col, (num, title, done) in zip(cols, outcomes):

        with col:

            mark = "✓" if done else "○"

            st.markdown(f'<div class="card"><span class="badge">{num}</span><h3>{mark} {title}</h3><div class="small">Academic expected outcome</div></div>', unsafe_allow_html=True)

    st.markdown("### Required deliverables")

    dcols = st.columns(5)

    for col, title in zip(dcols, ["Preprocessed corpus", "LDA model output", "Topic-word lists", "Interpretation report", "Visualizations"]):

        with col:

            st.markdown(f'<div class="card"><h4>✓ {title}</h4></div>', unsafe_allow_html=True)

    if not trained():

        st.info("Start with **Dataset** → **Preprocessing** → **Document-Term Matrix** → **LDA Configuration** → **Train Model**.")





def dataset_page():

    st.title("Dataset")

    mode = st.radio("Dataset source", ["Built-in BBC News Dataset", "Upload a document collection"], horizontal=True)

    if mode == "Built-in BBC News Dataset":

        st.caption("Real-world BBC News article collection. Category labels remain metadata and are excluded from LDA.")

        path = Path("data/bbc_news.csv")

        if st.button("Load BBC Dataset", type="primary"):

            try:

                df = load_bbc_csv(path)

                st.session_state.dataset = df

                st.session_state.dataset_name = "BBC News Dataset"

                reset_model_state()

                st.success(f"Loaded {len(df):,} documents.")

            except DatasetError as exc:

                st.error(str(exc))

        st.markdown("""

        <div class="card">

        <h3>BBC dataset setup</h3>

        <p>Place the real BBC dataset at <code>data/bbc_news.csv</code>, or use

        <code>python prepare_bbc_dataset.py</code> after extracting the standard BBC category folders.</p>

        <p class="small">LDA uses article text only. Category labels are not used as model features.</p>

        </div>

        """, unsafe_allow_html=True)

    else:

        upload = st.file_uploader("Upload CSV, TXT or JSON", type=["csv", "txt", "json"])

        if upload is not None and st.button("Use Uploaded Dataset", type="primary"):

            try:

                df = load_uploaded_file(upload)

                if len(df) < 2:

                    raise DatasetError("The uploaded collection must contain at least 2 documents.")

                st.session_state.dataset = df

                st.session_state.dataset_name = f"Uploaded: {upload.name}"

                reset_model_state()

                st.success(f"Loaded {len(df):,} documents. A new LDA model will be trained for this collection.")

            except DatasetError as exc:

                st.error(str(exc))

    df = st.session_state.dataset

    if df is not None:

        summary = dataset_summary(df)

        st.markdown("### Active dataset")

        c = st.columns(6)

        for col, (label, value) in zip(c, [

            ("Documents", f"{summary['documents']:,}"),

            ("Total words", f"{summary['total_words']:,}"),

            ("Avg length", f"{summary['average_length']:.1f}"),

            ("Min length", summary["minimum_length"]),

            ("Max length", summary["maximum_length"]),

            ("Columns", summary["columns"]),

        ]):

            with col:

                st.metric(label, value)

        st.dataframe(df.head(10), use_container_width=True)

        st.info("LDA is trained only on article text. Category labels, if present, are retained only as metadata.")





def preprocessing_page():

    st.title("Preprocessing")

    if st.session_state.dataset is None:

        st.warning("Load a dataset first.")

        return

    c1, c2, c3 = st.columns(3)

    with c1:

        remove_sw = st.checkbox("Stopword removal", True)

    with c2:

        lemmatize = st.checkbox("Lemmatization", True)

    with c3:

        min_len = st.slider("Minimum word length", 2, 8, 3)

    if st.button("Run Preprocessing", type="primary"):

        try:

            df, stats = add_preprocessed_column(

                st.session_state.dataset,

                remove_stopwords=remove_sw, lemmatize=lemmatize, min_word_length=min_len

            )

            if len(df) < 2:

                raise ValueError("Preprocessing removed too many documents. Relax the filters.")

            st.session_state.preprocessed = df

            st.session_state.preprocess_stats = stats

            st.session_state.preprocessing_config = {

                "stopword_removal": remove_sw,

                "lemmatization": lemmatize,

                "minimum_word_length": min_len,

            }

            st.session_state.vectorizer = st.session_state.dtm = None

            st.success("Preprocessing completed.")

        except Exception as exc:

            st.error(f"Preprocessing failed: {exc}")

    if st.session_state.preprocessed is not None:

        s = st.session_state.preprocess_stats

        cols = st.columns(4)

        for col, (label, value) in zip(cols, [

            ("Original words", f"{s['original_word_count']:,}"),

            ("Processed words", f"{s['processed_word_count']:,}"),

            ("Removed tokens", f"{s['removed_tokens']:,}"),

            ("Avg processed length", f"{s['average_document_length']:.1f}"),

        ]):

            col.metric(label, value)

        st.markdown("### BEFORE / AFTER")

        ex = st.session_state.preprocessed.head(5)

        for _, row in ex.iterrows():

            with st.expander(str(row["document_id"])):

                st.write("**Before**")

                st.write(str(row["text"])[:1200])

                st.write("**After**")

                st.write(str(row["preprocessed_text"])[:1200])





def dtm_page():
    st.title("Document-Term Matrix")

    if st.session_state.preprocessed is None:
        st.warning("Run preprocessing first.")
        return

    max_vocab = st.slider(
        "Maximum vocabulary size", 500, 20000, 5000, step=500
    )

    if st.button("Build Document-Term Matrix", type="primary"):
        try:
            vec, dtm = build_document_term_matrix(
                st.session_state.preprocessed["preprocessed_text"],
                min_df=2,
                max_df=0.90,
                max_features=max_vocab,
                ngram_range=(1, 1),
            )

            st.session_state.vectorizer = vec
            st.session_state.dtm = dtm
            st.session_state.dtm_summary = dtm_summary(dtm, vec)
            st.success("Document-term matrix created.")
        except Exception as exc:
            st.error(f"Could not build the document-term matrix: {exc}")

    if st.session_state.dtm is not None:
        s = st.session_state.dtm_summary

        cols = st.columns(4)
        for col, (label, value) in zip(cols, [
            ("Documents", f"{s['documents']:,}"),
            ("Vocabulary", f"{s['vocabulary_size']:,}"),
            ("Dimensions", f"{s['rows']} × {s['columns']}"),
            ("Sparsity", f"{s['sparsity']:.2%}"),
        ]):
            with col:
                col.metric(label, value)

        st.markdown("### Document-Term Matrix Preview")
        st.caption(
            f"The full matrix contains {s['rows']:,} documents × "
            f"{s['columns']:,} terms ({s['rows'] * s['columns']:,} cells). "
            "The controls below display a manageable preview; the complete "
            "matrix is retained internally and is used by the LDA model."
        )

        pcols = st.columns(2)

        with pcols[0]:
            preview_docs = st.slider(
                "Documents to display",
                min_value=5,
                max_value=min(50, s["rows"]),
                value=min(10, s["rows"]),
                step=5,
                key="dtm_preview_docs",
            )

        with pcols[1]:
            preview_terms = st.slider(
                "Terms to display",
                min_value=5,
                max_value=min(30, s["columns"]),
                value=min(15, s["columns"]),
                step=5,
                key="dtm_preview_terms",
            )

        feature_names = np.asarray(
            st.session_state.vectorizer.get_feature_names_out()
        )

        term_counts = np.asarray(
            st.session_state.dtm.sum(axis=0)
        ).ravel()

        informative_indices = np.argsort(term_counts)[::-1][:preview_terms]
        doc_indices = np.arange(min(preview_docs, s["rows"]))

        preview_array = st.session_state.dtm[
             doc_indices, :
        ][:, informative_indices].toarray()

        preview_df = pd.DataFrame(
            preview_array,
            index=[
                str(st.session_state.preprocessed.iloc[i]["document_id"])
                for i in doc_indices
            ],
            columns=feature_names[informative_indices],
        )

        preview_df.index.name = "Document"

        st.dataframe(
            preview_df,
            use_container_width=True,
        )

        st.caption(
            "The preview uses frequently occurring vocabulary terms so that "
            "the non-zero structure of the sparse matrix is easier to inspect."
        )

        st.markdown("### Explore an Individual Document")

        selected_doc = st.selectbox(
            "Select a document",
            range(len(st.session_state.preprocessed)),
            format_func=lambda i: str(
                st.session_state.preprocessed.iloc[i]["document_id"]
            ),
            key="dtm_document_selector",
        )

        selected_row = st.session_state.dtm[selected_doc]
        nonzero_indices = selected_row.indices
        nonzero_counts = selected_row.data

        if len(nonzero_indices) == 0:
            st.info(
                "This document has no terms in the current DTM vocabulary."
            )
        else:
            document_terms = pd.DataFrame({
                "Term": feature_names[nonzero_indices],
                "Count": nonzero_counts.astype(int),
            }).sort_values(
                ["Count", "Term"],
                ascending=[False, True],
            ).reset_index(drop=True)

            st.dataframe(
                document_terms,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                f"{len(document_terms):,} non-zero terms out of "
                f"{s['vocabulary_size']:,} vocabulary terms for the selected document."
            )

        st.info(
            "A zero means that the term is absent from that document. "
            "The vocabulary is shared across the entire corpus, so every "
            "document has the same set of columns. Text document-term "
            "matrices are naturally sparse because each document contains "
            "only a small fraction of the total vocabulary."
        )


def lda_config_page():

    st.title("LDA Configuration")

    if st.session_state.preprocessed is None:

        st.warning("Run preprocessing first.")

        return

    topics = st.slider("Number of topics", 2, 15, 5, help="Number of latent topics to discover.")

    iterations = st.slider("Maximum iterations", 10, 200, 50, step=10, help="Maximum optimization iterations.")

    method = st.selectbox("Learning method", ["batch", "online"], help="Batch uses the full dataset per update; online uses mini-batch-style updates.")

    random_state = st.number_input("Random state", value=42, step=1, help="Controls reproducibility.")

    top_words = st.slider("Top words per topic", 5, 20, 10)

    st.session_state.lda_config = {

        "n_topics": int(topics), "max_iter": int(iterations),

        "learning_method": method, "random_state": int(random_state),

        "top_words": int(top_words)

    }

    st.info("Changing parameters does not retrain the model. Training happens only from the Train Model page after you click the training button.")





def train_page():

    st.title("Train Model")

    if st.session_state.preprocessed is None:

        st.warning("Run preprocessing first.")

        return

    if st.session_state.dtm is None:

        st.warning("Build the document-term matrix first.")

        return

    if st.session_state.lda_config is None:

        st.warning("Configure LDA first.")

        return

    cfg = st.session_state.lda_config

    st.markdown(f'<div class="card"><h3>Ready to train</h3><p>{cfg["n_topics"]} topics · {cfg["max_iter"]} iterations · {cfg["learning_method"]} learning · random state {cfg["random_state"]}</p></div>', unsafe_allow_html=True)

    if st.button("Train LDA Model", type="primary", use_container_width=True):

        try:

            with st.status("Training LDA model…", expanded=True) as status:

                st.write("Using preprocessed corpus…")

                model = train_lda(

                    st.session_state.dtm,

                    n_topics=cfg["n_topics"], max_iter=cfg["max_iter"],

                    learning_method=cfg["learning_method"], random_state=cfg["random_state"]

                )

                st.write("Calculating topic-word distribution…")

                tw = get_topic_word_distribution(model)

                st.write("Calculating document-topic distribution…")

                dt = get_document_topic_distribution(model, st.session_state.dtm)

                st.write("Analyzing discovered topics and evaluation metrics…")

                summaries = topic_summary(

                    model, st.session_state.vectorizer, st.session_state.preprocessed,

                    dt, top_n=cfg["top_words"]

                )

                metrics = {

                    "log_likelihood": calculate_log_likelihood(model, st.session_state.dtm),

                    "perplexity": calculate_perplexity(model, st.session_state.dtm),

                }

                st.session_state.lda = model

                st.session_state.topic_word = tw

                st.session_state.doc_topic = dt

                st.session_state.topic_summaries = summaries

                st.session_state.metrics = metrics

                status.update(label="LDA model trained successfully", state="complete")

            st.success("✓ LDA model trained successfully")

        except Exception as exc:

            st.session_state.last_training_error = str(exc)

            st.error("Model training failed. Check the configuration and vocabulary size.")

            st.caption(str(exc))

    if trained():

        m = st.session_state.metrics

        cols = st.columns(6)

        vals = [

            ("Documents", len(st.session_state.preprocessed)),

            ("Vocabulary", st.session_state.dtm.shape[1]),

            ("Topics", st.session_state.lda.n_components),

            ("Iterations", st.session_state.lda.n_iter_),

            ("Log likelihood", f"{m['log_likelihood']:.2f}"),

            ("Perplexity", f"{m['perplexity']:.2f}"),

        ]

        for col, (label, value) in zip(cols, vals):

            col.metric(label, value)





def topic_discovery_page():

    st.title("Topic Discovery")

    if not require_training():

        return

    st.markdown("### What hidden topics did LDA discover in this document collection?")

    st.caption("Primary result: topics are discovered from word patterns across the entire corpus. Labels below are automatically generated and are not ground-truth categories.")

    for item in st.session_state.topic_summaries:

        with st.container():

            words = " ".join([f'<span class="word-chip">{w} · {p:.3f}</span>' for w, p in item["top_words"]])

            st.markdown(f"""

            <div class="topic-card">

              <div class="badge">TOPIC {item['topic']}</div>

              <div class="topic-label" style="margin-top:8px">{item['label']}</div>

              <div style="margin:12px 0">{words}</div>

              <p><b>Topic prevalence:</b> {item['prevalence']:.2%} &nbsp; <b>Representative/dominant documents:</b> {item['dominant_document_count']}</p>

              <p>{item['interpretation']}</p>

              <div class="small">Automatically generated interpretation — not a ground-truth label.</div>

            </div>

            """, unsafe_allow_html=True)

            with st.expander("Representative documents"):

                st.dataframe(item["representatives"], use_container_width=True, hide_index=True)





def topic_explorer_page():

    st.title("Topic Explorer")

    if not require_training():

        return

    n = st.session_state.lda.n_components

    selected = st.selectbox("Select discovered topic", range(1, n + 1))

    item = st.session_state.topic_summaries[selected - 1]

    st.subheader(f"Topic {selected} — {item['label']}")

    c1, c2 = st.columns([1.5, 1])

    with c1:

        st.plotly_chart(topic_word_bar(item["top_words"], selected), use_container_width=True)

        st.dataframe(

            pd.DataFrame(item["top_words"], columns=["Word", "Probability"]),

            use_container_width=True, hide_index=True

        )

    with c2:

        st.image(wordcloud_figure(item["top_words"]), caption="Word cloud from actual topic-word probabilities.")

        st.metric("Topic prevalence", f"{item['prevalence']:.2%}")

        st.metric("Dominant documents", item["dominant_document_count"])

    st.info(item["interpretation"])

    st.markdown("**Why this interpretation?** The label and explanation are generated directly from the highest-probability words shown above.")





def document_analysis_page():

    st.title("Document Analysis")

    if not require_training():

        return

    df = st.session_state.preprocessed

    idx = st.selectbox("Select a document", range(len(df)), format_func=lambda i: df.iloc[i]["document_id"])

    probs = st.session_state.doc_topic[idx]

    dominant = int(np.argmax(probs)) + 1

    c1, c2 = st.columns([1.4, 1])

    with c1:

        st.plotly_chart(document_topic_bar(probs, [f"Topic {i+1}" for i in range(len(probs))]), use_container_width=True)

    with c2:

        st.metric("Dominant topic", f"Topic {dominant}")

        st.metric("Dominant probability", f"{probs[dominant-1]:.2%}")

        st.dataframe(pd.DataFrame({"Topic": [f"Topic {i+1}" for i in range(len(probs))], "Probability": probs}), hide_index=True, use_container_width=True)

    st.markdown("### Article")

    st.write(df.iloc[idx]["text"])

    st.info("LDA represents each document as a mixture of the topics discovered across the corpus. A document is not permanently assigned to a single topic.")





def corpus_visualization_page():

    st.title("Corpus Visualization")

    if not require_training():

        return

    prevalence = topic_prevalence(st.session_state.doc_topic)

    st.plotly_chart(prevalence_chart(prevalence), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:

        st.plotly_chart(topic_similarity_heatmap(topic_similarity(st.session_state.topic_word)), use_container_width=True)

    with c2:

        st.plotly_chart(topic_document_heatmap(st.session_state.doc_topic), use_container_width=True)

    st.markdown("### Documents dominated by each topic")

    counts = pd.Series(st.session_state.doc_topic.argmax(axis=1) + 1).value_counts().sort_index()

    st.dataframe(pd.DataFrame({"Topic": counts.index, "Dominant documents": counts.values}), hide_index=True, use_container_width=True)





def evaluation_page():

    st.title("Model Evaluation")

    if not require_training():

        return

    m = st.session_state.metrics

    c1, c2 = st.columns(2)

    c1.metric("Log likelihood", f"{m['log_likelihood']:.2f}")

    c2.metric("Perplexity", f"{m['perplexity']:.2f}")

    st.info("Perplexity can help compare models, but a lower perplexity score does not necessarily mean that the topics are more interpretable.")

    counts = st.multiselect("Optional topic-count comparison", [2,3,4,5,6,7,8,10], default=[3,4,5,6,7])

    if st.button("Run Comparison"):

        with st.spinner("Training comparison models…"):

            result = compare_topic_counts(

                st.session_state.dtm, counts,

                max_iter=st.session_state.lda_config["max_iter"],

                learning_method=st.session_state.lda_config["learning_method"],

                random_state=st.session_state.lda_config["random_state"],

            )

        st.dataframe(result, use_container_width=True, hide_index=True)

        st.line_chart(result.set_index("Topics")[["Perplexity"]])

        st.line_chart(result.set_index("Topics")[["Log Likelihood"]])





def probabilistic_page():

    st.title("Probabilistic View")

    if not require_training():

        return

    st.markdown("### SECTION A — DOCUMENT → TOPIC")

    idx = st.selectbox("Document", range(len(st.session_state.preprocessed)), format_func=lambda i: st.session_state.preprocessed.iloc[i]["document_id"])

    probs = st.session_state.doc_topic[idx]

    st.plotly_chart(document_topic_bar(probs, [f"Topic {i+1}" for i in range(len(probs))]), use_container_width=True)

    st.dataframe(pd.DataFrame({"Topic": [f"Topic {i+1}" for i in range(len(probs))], "P(topic | document)": probs}), hide_index=True, use_container_width=True)

    st.markdown("### SECTION B — TOPIC → WORD")

    topic = st.selectbox("Topic", range(1, st.session_state.lda.n_components + 1))

    words = get_top_words_per_topic(st.session_state.lda, st.session_state.vectorizer, st.session_state.lda_config["top_words"])[topic-1]

    st.plotly_chart(topic_word_bar(words, topic), use_container_width=True)

    st.image(wordcloud_figure(words), caption="P(word | topic) visualized as a word cloud.")

    st.dataframe(pd.DataFrame(words, columns=["Word", "P(word | topic)"]), hide_index=True, use_container_width=True)

    st.info("P(topic | document) shows the probability that a document exhibits a discovered topic. P(word | topic) shows the probability of a word under a discovered topic. The charts and numbers above are real model output.")





def sampling_page():

    st.title("Sampling View")

    st.markdown("""

    <div class="card">

    <h3>Conceptual Gibbs Sampling Demonstration</h3>

    <p><b>Actual implementation:</b> scikit-learn LatentDirichletAllocation using variational inference.</p>

    <p><b>Conceptual alternative:</b> Gibbs sampling for LDA inference.</p>

    </div>

    """, unsafe_allow_html=True)

    step = st.slider("Walk through the conceptual process", 1, 4, 1)

    steps = {

        1: ("Latent topic assignments", "Imagine each word token has a latent topic assignment that is initially unknown."),

        2: ("Word-topic assignments", "Group token assignments by topic and track how words occur across topics."),

        3: ("Iterative conditional updates", "Repeatedly update a token's topic using conditional probabilities based on topic-word and document-topic counts."),

        4: ("Convergence", "After many iterations, the topic assignments can stabilize into useful latent themes and topic-word distributions."),

    }

    title, desc = steps[step]

    st.markdown(f"### Step {step}: {title}")

    st.info(desc)

    st.caption("This is an educational conceptual view. TopicLens does not claim that sklearn's LDA backend performs Gibbs sampling.")





def reports_page():

    st.title("Reports & Downloads")

    if not require_training():

        return

    state = {

        "dataset_name": st.session_state.dataset_name,

        "dataset_summary": dataset_summary(st.session_state.dataset),

        "preprocessing_config": st.session_state.preprocessing_config,

        "preprocess_stats": st.session_state.preprocess_stats,

        "dtm_summary": st.session_state.dtm_summary,

        "lda_config": st.session_state.lda_config,

        "metrics": st.session_state.metrics,

        "topic_summaries": st.session_state.topic_summaries,

    }

    report = generate_report(state)

    topic_df = topic_word_dataframe(st.session_state.lda, st.session_state.vectorizer, st.session_state.lda_config["top_words"])

    doc_df = document_topic_dataframe(st.session_state.lda, st.session_state.dtm, st.session_state.preprocessed)

    prep_df = st.session_state.preprocessed.copy()

    st.markdown("### Required deliverables")

    c1, c2 = st.columns(2)

    with c1:

        st.download_button("Download preprocessed_corpus.csv", prep_df.to_csv(index=False), "preprocessed_corpus.csv", "text/csv")

        st.download_button("Download topic_word_distribution.csv", topic_df.to_csv(index=False), "topic_word_distribution.csv", "text/csv")

        st.download_button("Download document_topic_distribution.csv", doc_df.to_csv(index=False), "document_topic_distribution.csv", "text/csv")

    with c2:

        summary_txt = json.dumps({

            "dataset": st.session_state.dataset_name,

            "lda_configuration": st.session_state.lda_config,

            "metrics": st.session_state.metrics,

            "topic_distributions": st.session_state.topic_word.tolist(),

        }, indent=2)

        st.download_button("Download LDA model output", summary_txt, "lda_model_output.txt", "text/plain")

        st.download_button("Download Topic Interpretation Report", report, "topic_interpretation_report.txt", "text/plain")

    st.markdown("### Report preview")

    st.text_area("Generated report", report, height=500)





def main():

    init_state()

    st.sidebar.markdown("## ◈ TopicLens")

    st.sidebar.caption("Probabilistic topic modeling dashboard")

    if st.session_state.dataset_name:

        st.sidebar.markdown(f"**ACTIVE DATASET**  \n\`{st.session_state.dataset_name}\`")

    else:

        st.sidebar.markdown("**ACTIVE DATASET**  \n\`Not selected\`")

    status_badge()

    page = st.sidebar.radio("Navigate", PAGES)

    st.sidebar.divider()

    st.sidebar.caption("LDA discovers hidden topics from a collection of documents.")

    dispatch = {

        "Overview": overview,

        "Dataset": dataset_page,

        "Preprocessing": preprocessing_page,

        "Document-Term Matrix": dtm_page,

        "LDA Configuration": lda_config_page,

        "Train Model": train_page,

        "Topic Discovery": topic_discovery_page,

        "Topic Explorer": topic_explorer_page,

        "Document Analysis": document_analysis_page,

        "Corpus Visualization": corpus_visualization_page,

        "Model Evaluation": evaluation_page,

        "Probabilistic View": probabilistic_page,

        "Sampling View": sampling_page,

        "Reports": reports_page,

    }

    dispatch[page]()





if __name__ == "__main__":

    main()
