# Measuring Corporate Culture Using Machine Learning — Chinese Adaptation 中文版

## 📘 Introduction

This repository implements a **Chinese-adapted version** of the method proposed in the paper:

> Kai Li, Feng Mai, Rui Shen, Xinyan Yan,  
> [__Measuring Corporate Culture Using Machine Learning__](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhaa079/5869446?redirectedFrom=fulltext),  
> _The Review of Financial Studies_, 2020.  
> DOI: [10.1093/rfs/hhaa079](http://dx.doi.org/10.1093/rfs/hhaa079)

The original implementation is designed for **English-language corporate texts** (e.g., earnings calls).  
This repository extends the pipeline to support **Chinese textual data**, including Chinese annual reports, CSR reports, and management discussions.

本项目基于上述论文方法进行了中文化改造，实现了从中文语料的预处理、分词、短语学习、到词向量建模与得分计算的完整流程。

The modified pipeline has been tested on **Windows 10**, **Ubuntu 20.04**, and **macOS Sonoma**,  
using Python 3.8–3.12 environments.

---

## 🧩 Key Modifications for Chinese Version

| Component | Original (English) | Modified (Chinese) |
|------------|--------------------|--------------------|
| Tokenization | Stanford CoreNLP (English POS/NER) | Stanford CoreNLP (Chinese POS/NER), optional **pkuseg** / **THULAC** |
| Stopwords | NLTK stopword list | Chinese stopword dictionary (`data/resources/StopWords_Generic.txt`) |
| Named Entity Removal | CoreNLP NER | Optional filtering using **HanLP** or **pkusegNER** |
| Phrase Detection | gensim Phrases (2–3 gram) | Retained; applied on tokenized Chinese words |
| Word Embedding | gensim Word2Vec | Compatible; trained on Chinese bigram/trigram corpus |
| Dictionary Expansion | cosine similarity in embedding space | Fully retained |
| Scoring | TF / TF-IDF / WF-IDF | Fully retained, adapted for Chinese vocabulary |
| Optional Aggregation | by firm–year | Fully retained |

---

## ⚙️ Requirements

- Python 3.8+
- Install required packages:
  ```bash
  pip install -r requirements.txt
 ```

### Stanford CoreNLP

If you still wish to use **Stanford CoreNLP** (e.g., for dependency parsing or POS tagging in English–Chinese mixed texts),
download [Stanford CoreNLP v4.5.4+](https://nlp.stanford.edu/software/stanford-corenlp-4.5.10-models-chinese.jar)
and specify its path in `global_options.py`:

```python
os.environ["CORENLP_HOME"] = "/your/path/stanford-corenlp-4.5.10/"
```
---

## 📁 Data Format

Example input files should be placed in `data/input/`:

| File                        | Description                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `documents.txt`             | Each line is a full document (e.g., one annual report, CSR report, or MD&A text).  |
| `document_ids.txt`          | Each line is a unique document ID (e.g., firm–year).                              |
| `id2firms.csv` *(optional)* | CSV with columns: `document_id` (str), `firm_id` (str), `time` (int). Used for firm–time aggregation.           |

---

## ⚙️ Global Options

Edit `global_options.py` to set:

* RAM and CPU cores for training and parsing
* Stopword file path (`data/resources/StopWords_Generic.txt`)
* Seed word list (see `data/dictionaries/seed_words.csv`)
* Max number of words per cultural dimension
* Whether to enable bigram/trigram detection

---

## 🚀 Running the Code

### 1️⃣ Text Parsing / Tokenization

Run:

```bash
python parse.py
```

This step performs **Chinese word segmentation**, **optional NER removal**, and saves segmented results to:

```
data/processed/parsed/
```

Output files:

* `documents_tokenized.txt`: segmented sentences (one per line)
* `document_sent_ids.txt`: document–sentence mapping IDs

---

### 2️⃣ Cleaning and Training Word2Vec

Run:

```bash
python clean_and_train.py
```

This script:

* Removes stopwords and numerics
* Learns bigrams and trigrams using `gensim.models.Phrases`
* Trains a Chinese Word2Vec model

Outputs:

* `models/w2/w2v.mod`
* `models/phrases/bigram.mod`
* `models/phrases/trigram.mod`

---

### 3️⃣ Dictionary Expansion

Run:

```bash
python create_dict.py
```

It expands the seed dictionary (e.g., integrity, teamwork, innovation, respect, and customer orientation)
using cosine similarity in the embedding space.

Output:

```
outputs/dict/expanded_dict.csv
```

---

### 4️⃣ Scoring Documents

Run:

```bash
python score.py
```

Generates culture scores for each document using three schemes:

* **TF** (term frequency)
* **TF-IDF**
* **WF-IDF** (log-scaled TF-IDF)

Outputs:

```
outputs/scores/scores_TF.csv
outputs/scores/scores_TFIDF.csv
outputs/scores/scores_WFIDF.csv
```

---

### 5️⃣ (Optional) Aggregation

Aggregate to firm–year level:

```bash
python aggregate_firms.py
```

Output:

```
outputs/scores/firm_scores_TF.csv
outputs/scores/firm_scores_TFIDF.csv
outputs/scores/firm_scores_WFIDF.csv
```

---

## 📚 Notes and Tips

* Recommended to clean Chinese input files with UTF-8 encoding (remove HTML tags, control characters).
* For large corpora, training can be parallelized by increasing `workers` in `global_options.py`.
* You may extend the seed dictionary to domain-specific dimensions (e.g., ESG, innovation, compliance).

---

## 🧠 Citation

If you use this repository, please cite the original paper:

> Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan.
> *Measuring Corporate Culture Using Machine Learning.*
> *The Review of Financial Studies*, 2020.
> DOI: [10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079)

And please mention the Chinese adaptation (this repository).

## 🧠 Example Workflow

```bash
python parse.py
python clean_and_train.py
python create_dict.py
python score.py
python aggregate_firms.py
```

After running, check:

* `outputs/scores/scores_TF.csv` → document-level scores
* `outputs/scores/scores_TFIDF.csv` → document-level scores
* `outputs/scores/scores_WFIDF.csv` → document-level scores
* `outputs/scores/firm_scores_TF.csv` → firm–year aggregated indices
* `outputs/scores/firm_scores_TFIDF.csv` → firm–year aggregated indices
* `outputs/scores/firm_scores_WFIDF.csv` → firm–year aggregated indices

---

## 🧩 Acknowledgement

This Chinese adaptation builds upon the original RFS 2020 paper and open-source code.
All modifications are intended to enable cross-lingual and domain adaptation for Chinese corporate texts.

---

### 🔖 License

This project is distributed under the same license as the original repository (see `LICENSE` file).
Please attribute the original authors and this adaptation appropriately.

```

