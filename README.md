# Understanding How News Articles Shape Public Opinion

**A Comprehensive Machine Learning Pipeline for Media Framing and Sentiment Analysis**

**Department:** School of Computing Science and Artificial Intelligence, VIT Bhopal University
**Author:** Rananjay Singh Chauhan (23BAI10080)  
**Supervisor:** Dr. Manorama Chouhan (manoramachouhan@vitbhopal.ac.in)

---

## 📖 What We Are Actually Trying to Do

Standard sentiment analysis is built for product reviews and tweets — text that wears its feelings on its sleeve. News is different. A BBC dispatch doesn't say "I'm sad about this." It says "the minister *admitted*" instead of "the minister *said*." The emotion is in the verb choice, the adjective selection, the source quoted, and the entity framed. That's framing — and framing is what this system detects.

The system takes a news article (text or broadcast transcript), a named entity within it, and outputs one of four labels:

- **Supportive** — the entity is portrayed positively
- **Critical** — the entity is blamed, questioned, condemned
- **Neutral-Reporting** — wire-service register; factual, balanced
- **Alarmist** — crisis/urgency framing regardless of strict valence

We do this across a diverse selection of outlets (listed in `configs/outlets.json`).
Then we ask: does framing diverge by outlet? By topic? By entity type?  
**That's the research question. The code answers it empirically.**

---

## 🏗️ Architecture Overview

The pipeline is modular by design, with five independently executable stages, orchestrated by `main.py`:

```text
news_sentiment/
├── main.py                     ← orchestrator: runs all modules in sequence
├── configs/
│   ├── outlets.json            ← outlet names, RSS URLs, scrape strategies
│   ├── model_config.yaml       ← hyperparameters, paths, label maps
│   └── env_config.py           ← path handling and configuration schema
│
├── data_collection/            ← MODULE 1
│   ├── scraper.py              ← newspaper3k + retry logic
│   ├── rss_collector.py        ← RSS feed parser per outlet
│   ├── fallback_loader.py      ← CC-News / MIND / SemEval datasets
│   ├── deduplicator.py         ← SHA-256 content hashing
│   └── schema.py               ← Article dataclass + validators
│
├── preprocessing/              ← MODULE 2
│   ├── cleaner.py              ← HTML strip, unicode norm, boilerplate removal
│   ├── ner_pipeline.py         ← spaCy en_core_web_trf, entity extraction
│   ├── proximity_scorer.py     ← syntactic distance via dep parse tree
│   └── asr_cleaner.py          ← Whisper transcript noise removal
│
├── models/                     ← MODULE 3
│   ├── baselines.py            ← LR, SVM, Naive Bayes (sklearn)
│   ├── roberta_framing.py      ← RoBERTa fine-tuner (HuggingFace)
│   ├── entity_attention.py     ← Entity-aware attention layer
│   └── fusion.py               ← Late fusion: text + ASR transcript
│
├── evaluation/                 ← MODULE 4
│   ├── metrics.py              ← Macro-F1, confusion matrix, per-class
│   ├── statistical_tests.py    ← Mann-Whitney U, Kruskal-Wallis, ANOVA
│   └── kappa.py                ← Fleiss' Kappa, Cohen's Kappa
│
└── analysis/                   ← MODULE 5
    ├── cross_source.py         ← Outlet × Topic framing heatmaps
    └── entity_profiler.py      ← Per-entity framing profiles
```

---

## 🗃️ Data Collection

### What it does

1. Reads `configs/outlets.json` — 10 outlets, each with RSS URL(s) and a scrape strategy flag.
2. For each outlet: parse RSS → extract article URLs → scrape full text via `newspaper3k`.
3. On any failure (403, timeout, parse error, empty body): falls back to public datasets.
4. Deduplicates using SHA-256 hash of the article body.
5. Validates and writes to `data/raw/{outlet_name}.jsonl`.

### Fallback datasets

| Dataset       | Use case                          | Access method             |
|---------------|-----------------------------------|---------------------------|
| MIND          | Topic diversity, entity density   | `datasets` library (HF)   |
| CC-News       | International outlet diversity    | `datasets` library (HF)   |
| SemEval-2017  | Entity-level gold labels          | Manual download + loader  |

### Article schema (every record written to JSONL)

```python
@dataclass
class Article:
    article_id: str        # SHA-256 of body (first 16 chars)
    source: str            # outlet name
    title: str             # headline
    body: str              # full article text
    url: str               # canonical URL
    date: str              # ISO 8601
    topic: str             # auto-inferred from RSS category
    entities: list[str]    # empty at collection time; filled in Module 2
    label: str             # empty at collection time; filled in annotation
    transcript: str        # empty unless ASR — filled in Module 2
```

The Data Collection module is the ingestion engine for the project. Its primary responsibility is to gather raw, unstructured news articles from various media outlets across the globe and normalize them into a structured, consistent dataset ready for text preprocessing and machine learning analysis.

## Role and Importance
Robust machine learning relies entirely on the quality and volume of its training data. This module ensures that our sentiment and framing classifiers are fed high-quality, real-world data directly from the source. By handling network timeouts, bot-blocking measures, pagination, and data schema enforcement, this module guarantees that the downstream pipeline always receives clean, structured JSONL files.

## Files and Workflow

### 1. `rss_collector.py`
- **Purpose:** Discovers and parses recent article URLs from the configured RSS feeds.
- **How it works:** It utilizes the `feedparser` library to read XML feeds defined in `outlets.json`. It extracts the article links, publication dates, and titles, returning a list of targets for the scraper.

### 2. `scraper.py`
- **Purpose:** Downloads the full HTML body of the articles and extracts the core textual content.
- **How it works:** Using `requests` and `BeautifulSoup4` (or `newspaper3k` concepts), it navigates to the URLs identified by the `rss_collector`. It extracts the primary article text while stripping out ads, navigation bars, and boilerplate HTML. It incorporates polite scraping practices (rate limiting, user-agent rotation) to prevent IP bans.

### 3. `fallback_loader.py`
- **Purpose:** Provides supplementary data when live scraping fails or yields insufficient articles.
- **How it works:** If an outlet completely blocks our scraper or experiences an outage (yielding 0 articles), this script uses the HuggingFace `datasets` library to stream backup articles from established public datasets (such as CC-News, MIND, or SemEval). This ensures our pipeline never halts due to unexpected network errors.

### 4. `deduplicator.py`
- **Purpose:** Prevents duplicate articles from polluting the dataset across multiple runs.
- **How it works:** It generates unique hashes for each article based on its URL and content. These hashes are checked against a persistent local store (`seen_ids.txt`). Any duplicates are dropped before writing to disk.

### 5. `schema.py`
- **Purpose:** Enforces a rigid data structure for all collected articles.
- **How it works:** Defines the `Article` Pydantic model, ensuring every piece of data has a `title`, `body`, `source`, `url`, `date`, `topic`, and `region`. It also contains heuristics to automatically infer an article's topic based on keywords.

### 6. `writer.py`
- **Purpose:** The orchestrator of the data collection phase. 
- **How it works:** Contains `run_collection()`, which acts as the main entry point for the module. It sequentially calls the collector, scraper, fallback loader, and deduplicator. Finally, it writes the validated `Article` objects into line-delimited JSON (`.jsonl`) files in the raw data directory, ready for the preprocessing stage.

---

# Module 2: Preprocessing & Feature Extraction

This module transforms raw, unstructured scraped articles and noisy ASR broadcast transcripts into clean, structured data ready for machine learning models. 

Text normalization is inherently messy. This stage normalizes the data by removing HTML noise, handling punctuation, extracting named entities, and calculating the syntactic distance between tokens and target entities.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `cleaner.py` | Performs foundational cleaning: HTML stripping, unicode normalization, boilerplate removal, and whitespace standardization. |
| `ner_pipeline.py` | Uses `spaCy` (transformer-backed `en_core_web_trf`) to perform Named Entity Recognition (NER). It identifies and extracts entities like PERSON, ORG, GPE, EVENT, and NORP. |
| `proximity_scorer.py` | Analyzes the dependency parse tree to compute the "syntactic distance" between evaluative words (adjectives/verbs) and the target entity. This is crucial for entity-centric framing, ensuring the model doesn't falsely attribute document-level sentiment to the specific entity. |
| `asr_cleaner.py` | Cleans and normalizes noisy Automated Speech Recognition (ASR) transcripts generated via OpenAI's `Whisper` model, handling disfluencies and speaker artifacts. |

## ⚙️ How it Fits into the Pipeline

1. **Input:** Receives raw JSONL files from the Data Collection module (`data/raw/`).
2. **Processing:** Cleans the text, runs it through the spaCy transformer pipeline to identify target entities, and calculates proximity scores.
3. **Output:** Saves the processed articles to `data/processed/`, which are then fed into the Machine Learning models in Module 3.

---

# Module 3: Machine Learning Models

This module contains the core analytical and classification logic. It ingests the preprocessed, structurally quantified text (along with its calculated syntactic entity-proximity scores) and outputs the predicted rhetorical framing label for the target entity.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `baselines.py` | Implements traditional Machine Learning baselines for comparison, utilizing `scikit-learn`. Includes standard models like Logistic Regression (LR), Support Vector Machines (SVM), and Naive Bayes, along with TF-IDF vectorization. |
| `roberta_framing.py` | The main powerhouse. A fine-tuning script utilizing HuggingFace's `transformers` library to adapt the `roberta-base` encoder. It uses a custom 4-class framing classification head and a class-weighted cross-entropy loss function optimized via AdamW. |
| `entity_attention.py` | A custom architectural layer injected between the RoBERTa encoder and the classification head. It forces the model to attend specifically to the target entity and its immediate syntactic context, preventing the model from lazily assigning the general document sentiment to the entity. |
| `fusion.py` | Implements "Late Fusion" mechanics. It allows the model to intelligently merge predictions/features from both the written text article and the ASR broadcast transcript when analyzing multimodal outlets (like BBC or CNN broadcast clips). |

## ⚙️ How it Fits into the Pipeline

1. **Input:** Receives clean, preprocessed data and target entities from Module 2.
2. **Processing:** Passes the data through the entity-aware RoBERTa model to classify the framing into one of four categories: *Supportive, Critical, Neutral-Reporting, or Alarmist*.
3. **Output:** Predicted labels are appended to the dataset, ready for statistical evaluation and analysis in Modules 4 and 5.

---

# Module 4: Evaluation

This module is responsible for rigorously quantifying the performance of the Machine Learning models from Module 3 and determining the inter-rater reliability of the weak-supervision annotation process.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `metrics.py` | Calculates standard classification metrics to evaluate the model's predictive power. This includes Macro-F1 score, Precision, Recall, and the generation of detailed Confusion Matrices to track misclassifications between labels like *Critical* and *Alarmist*. |
| `statistical_tests.py` | Crucial for the research thesis. It runs tests like Mann-Whitney U, Kruskal-Wallis, and ANOVA to determine if the divergence in framing across different media outlets is statistically significant, or just random noise. |
| `kappa.py` | A custom implementation of Inter-Rater Reliability (IRR) metrics. It calculates Fleiss' Kappa (for multi-rater agreement) and Cohen's Kappa to validate the quality of the silver-standard annotations generated by the `auto_annotate` system. |

## ⚙️ How it Fits into the Pipeline
It ingests the predictions generated by `models/` and outputs quantifiable metrics and p-values, proving the validity of both the deep learning architecture and the final framing divergence claims.

---

# Module 5: Analysis & Visualization

This is the final stage of the pipeline. While Module 4 handles the raw mathematics of model evaluation, Module 5 handles the interpretation and visualization of the results, specifically aimed at answering the core research questions regarding media framing bias.

## 📁 File Structure & Responsibilities

| File | Purpose |
|------|---------|
| `cross_source.py` | Generates Outlet × Topic framing heatmaps using `seaborn` and `matplotlib`. This script visualizes exactly how different international outlets (like BBC vs. RT vs. Firstpost) diverge when reporting on identical topics (like Politics or Conflict). |
| `entity_profiler.py` | Generates framing profiles for specific named entities. It aggregates the data to show whether a specific person or organization is consistently framed favorably (Supportive) or negatively (Critical/Alarmist) across the global media spectrum. |

## ⚙️ How it Fits into the Pipeline
It ingests the final classified dataset and outputs high-quality, IEEE-standard graphs, heatmaps, and distribution plots into the `data/` and `logs/` directories, serving as the visual evidence for the research paper.

---

## ⚙️ Tech Stack — Every Component Explained

### Language and Runtime
**Python 3.10+** — 3.12 in the sandbox, 3.10 on Kaggle's default kernel.  
Why not 3.13? transformers and torch lag on new Python versions. 3.10–3.12 is the safe window for the entire stack.

### Environment Detection
The code detects at runtime whether it's running on Kaggle (checks `/kaggle/input`) or an IDE (falls back to local paths). This sets:
- Data paths (Kaggle: `/kaggle/working/`, local: `./data/`)
- Device (`cuda` if available, `mps` if Apple Silicon, else `cpu`)
- Batch sizes (Kaggle T4: 16, local CPU: 4)
- Logging verbosity

### Data Collection Layer
- **`newspaper3k==0.2.8`**: Downloads and parses article HTML, extracting the core text while ignoring boilerplate. 
- **`feedparser==6.0.12`**: Parses RSS feeds. This is used before newspaper3k to cleanly gather article URLs and summaries.
- **`requests==2.33.1`**: HTTP client with retry logic (exponential backoff) and rotating User-Agent headers to reduce 403 blocks.
- **`datasets` (HuggingFace)**: Streams fallback datasets like MIND and CC-News when direct web scraping fails or yields zero articles.

### Deduplication
- Uses **SHA-256 hashes** of the article body. Any article whose hash exists in the seen-set is skipped.

### Storage Format
- **JSONL (JSON Lines)**: Data is stored with one JSON object per line, ensuring robust portability and immediate readiness for Pandas and PyTorch.

### Preprocessing Layer (Module 2 preview)
- **`spaCy` (en_core_web_trf)**: Transformer-backed NER, identifying PERSON, ORG, GPE, EVENT, NORP entities.
- **`Whisper` (openai-whisper)**: ASR for broadcast transcripts.

### Modelling Layer (Module 3 preview)
- **`transformers`**: `roberta-base` as the primary encoder, fine-tuned with a 4-class framing head and entity-aware attention.
- **`torch`**: Training loop, optimizer (AdamW), scheduler (linear warmup + cosine decay), class-weighted cross-entropy loss.
- **`scikit-learn`**: Training loops, baseline classifiers, and TF-IDF vectorization.

### Evaluation Layer (Module 4 preview)
- **`scipy.stats`**: Mann-Whitney U, Kruskal-Wallis H-test for outlet divergence significance.
- **`sklearn.metrics`**: F1, precision, recall, confusion matrix.
- **`statsmodels`**: ANOVA, post-hoc Tukey HSD.
- **Custom Fleiss' Kappa**: written from scratch.

### Analysis Layer (Module 5 preview)
- **`seaborn` + `matplotlib`**: outlet × topic framing heatmaps, entity framing profiles, distribution plots.
- **`pandas`**: all aggregation and groupby operations.

---

## 🛠️ Coding Conventions

- **Type hints everywhere** — Python 3.10+ union syntax (`str | None`)
- **Dataclasses** for all data structures — no raw dicts passed between modules
- **Logging** via `logging` stdlib — not print statements. Level set by env.
- **Docstrings** — Google style, one per function
- **No global state** — config passed explicitly; nothing imported from `__main__`
- **Fail loudly** — exceptions are caught, logged, and re-raised with context. No silent swallowing.
- **Reproducibility** — all random seeds set via `utils.seed_everything(seed=42)`

---

## ❌ What This Is Not

- Not a real-time system. This is a batch research pipeline.
- Not a production scraper. We respect `robots.txt` and rate limits.
- Not claiming the model is objective truth. Framing detection is linguistic measurement, not fact-checking.

---

## ✨ Recent System Enhancements & Production Features

### 1. True Entity-Centric Framing Analysis
- **Syntactic Proximity Scoring ($s_i$)**: Aligned real subword token proximity decay scores ($s_i = \frac{1}{1 + d(i, E)}$) with RoBERTa's `EntityAwareAttention` layer ($\alpha_i = \text{softmax}(W_e h_i + W_s s_i)$), allowing the attention head to attend specifically to words surrounding the target entity.
- **Entity-Focused Context Windowing**: Scans full article texts to extract paragraphs specifically mentioning the target entity rather than relying on generic article intros.
- **Entity-Anchored Hypotheses**: Formulates target-bound zero-shot prompts: `"In this news text, the target entity 'X' is portrayed in a way that is {}"`.
- **Cross-Entity Framing Comparison**: Interactive web interfaces generate comparative framing breakdowns for all prominent entities detected in an article.

### 2. Dual Web Application Support
- **Interactive Local Console (`webapp/`)**: Full-featured UI running on port 8000 with GPU support, custom RoBERTa inference, BART-large-mnli zero-shot ensemble, and real-time MLflow experiment tracking (`mlflow.db`).
- **Render Cloud Deployment (`webapp_render/`)**: Lightweight deployment optimized for Render Free Tier (512MB RAM limit) using CPU DistilBERT and declarative `render.yaml`.

### 3. Pipeline Resilience & Fault Tolerance
- **403 Scraping Impersonation**: Uses `curl_cffi` with Chrome 120 TLS fingerprinting to bypass anti-bot scrapers.
- **Auto-Annotation Fallback Ladder**: Graceful degradation under memory pressure (`BART-Large GPU` $\rightarrow$ `DistilBERT GPU` $\rightarrow$ `DistilBERT CPU`).
- **Imbalanced Class Split Handling**: Reverts to unstratified splits if rare framing classes contain $<2$ samples, preventing baseline training crashes.

---

## 🏃 Sprint Plan

| Sprint | Module | Deliverable |
|--------|--------|-------------|
| 1 (now)| Data Collection | `data_collection/` — fully tested, dual-env |
| 2      | Preprocessing | `preprocessing/` — spaCy pipeline + ASR cleaner |
| 3      | Modelling | `models/` — baselines + RoBERTa fine-tuner |
| 4      | Evaluation | `metrics/` — all statistical tests |
| 5      | Analysis | `analysis/` — heatmaps + entity profiles |
| 6      | Integration | `main.py` + end-to-end test on 100 articles |

---

## 🚀 How to Run

1. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute the Batch Pipeline:**  
   ```bash
   python main.py
   ```
3. **Launch Local Interactive WebApp:**  
   ```bash
   uvicorn webapp.server:app --host 127.0.0.1 --port 8000 --reload
   ```

*(Note: Data is automatically saved into the `data/` directory, which is excluded from version control to protect data privacy and reduce repository bloat).*

---

## 📬 Contact & Contributions

For academic inquiries, peer reviews, or collaboration regarding the methodologies used in this study, please reach out to:  

**Rananjay Singh Chauhan**  
✉️ **Email:**  [rjchauhan.work@gmail.com](mailto:rjchauhan.work@gmail.com) | [rananjaychauhan93@gmail.com](mailto:rananjaychauhan93@gmail.com) | [rananjay.23bai10080@vitbhopal.ac.in](mailto:rananjay.23bai10080@vitbhopal.ac.in) 

🔗 **LinkedIn:** [linkedin.com/in/maihun-rsc](https://www.linkedin.com/in/maihun-rsc/) 

💻 **GitHub:** [github.com/maihun-rsc](https://github.com/maihun-rsc)  

