# 项目 Code Wiki

## 1. 项目概述 (Project Overview)

本项目是基于论文《Measuring Corporate Culture Using Machine Learning》的中文化适配版本，旨在通过机器学习与自然语言处理（NLP）技术，从中文企业文本（如年报、CSR报告、管理层讨论与分析 MD&A 等）中测量并量化企业的文化得分（如“绿色披露”、“创新”、“诚信”等维度）。

项目的核心流程包括：中文语料的预处理与分词、N-gram 短语学习、Word2Vec 词向量训练、基于种子词的字典扩展（Dictionary Expansion），以及最终基于 TF/TF-IDF/WF-IDF 的企业文化得分计算。

---

## 2. 项目整体架构 (Architecture)

本项目的架构按照数据流向和处理阶段可以分为以下几个核心层：

1. **配置层 (Configuration)**
   - 全局控制软硬件资源分配、目录路径、模型超参数（如 Word2Vec 维度）、停用词以及**种子词典**。
2. **数据预处理与分词层 (Data Preprocessing & Tokenization)**
   - 负责读取原始的文本文件（`documents.txt`），执行中文分词、去停用词、去除标点和数字，以及可选的命名实体识别（NER）。
3. **模型训练层 (Model Training)**
   - 在清洗后的单词（unigram）基础上，利用 Gensim 训练 Bigram（二元词组）和 Trigram（三元词组）模型。
   - 基于 Trigram 语料训练 Word2Vec 词向量模型，捕捉词汇的语义和上下文关联。
4. **字典扩展层 (Dictionary Expansion)**
   - 读取配置中的种子词（如：“环保”、“绿色”等），在词向量空间中利用余弦相似度寻找最相近的词，扩展并构建完整的“文化特征词典”。
5. **评分与聚合层 (Scoring & Aggregation)**
   - 扫描所有文档，计算扩展词典中词汇在各文档中的 TF、TF-IDF 或 WF-IDF 得分。
   - 将文档级别的得分通过 ID 映射表，聚合并平均到“公司-年度” (Firm-Year) 级别。

---

## 3. 主要模块与目录职责 (Module Responsibilities)

### 3.1 根目录脚本 (Root Scripts)
这些是执行流水线的入口脚本：
- **`global_options.py`**: 全局配置文件。配置项目路径、Stanford CoreNLP 路径、CPU核心数、Word2Vec 参数（维度、窗口大小）、种子词（`SEED_WORDS`）等。
- **`parse.py` / `jieba_tokenize.py`**: 第一步，分词与解析。`jieba_tokenize.py` 提供了快速的 Jieba 分词替代方案，而 `parse.py` 则倾向于使用 Stanford CoreNLP。
- **`clean_and_train.py`**: 第二步，清洗文本并训练模型。生成 N-gram 语料库，并训练 Gensim Word2Vec 模型。
- **`create_dict.py`**: 第三步，构建并扩展文化词典。输出扩展后的词汇表（CSV 格式）。
- **`score.py`**: 第四步，文档评分。根据扩展词典计算每个文档的文化特征分数。
- **`aggregate_firms.py`**: 第五步，数据聚合。将 `score.py` 的结果根据 `id2firms.csv` 聚合至公司-年份层面。

### 3.2 `culture/` 核心逻辑模块
存放了项目底层的类与函数实现：
- **`culture_dictionary.py`**: 负责词典扩展（相似度计算、去重）以及文档打分（TF、TF-IDF 算法实现）。
- **`culture_models.py`**: 封装了 Gensim 的短语模型（Phrases）与 Word2Vec 模型的训练及保存逻辑。
- **`preprocess.py` & `preprocess_parallel.py`**: 包含对 Stanford CoreNLP Client 的封装类，处理依赖解析、MWE（多词表达式）拼接及 NER 过滤等高级文本清洗逻辑。
- **`file_util.py`**: 文件读写及行数统计等基础工具类。

### 3.3 数据与输出目录 (Data & Outputs)
- **`data/input/`**: 存放用户输入文件（`documents.txt`, `document_ids.txt`, `id2firms.csv`）。
- **`data/processed/`**: 存放中间处理结果（分词后的文本、unigram/bigram/trigram 语料）。
- **`data/resources/`**: 存放外部资源（如中文停用词表 `StopWords_Generic.txt`）。
- **`models/`**: 保存训练好的 Gensim Word2Vec 模型和 Phraser 短语模型。
- **`outputs/`**: 存放最终输出结果，包括扩展词典 (`dict/`) 和计算得出的打分表 (`scores/`)。

---

## 4. 关键类与函数说明 (Key Classes and Functions)

### 4.1 `culture.culture_dictionary` 模块
- **`expand_words_dimension_mean(word2vec_model, seed_words, ...)`**
  - **功能**: 给定种子词，计算维度的平均词向量，并利用 `word2vec_model.wv.most_similar` 在词表空间中搜索距离最近的 $N$ 个词汇。
  - **返回**: 扩展后的维度词典（Dict）。
- **`deduplicate_keywords(word2vec_model, expanded_words, seed_words)`**
  - **功能**: 处理交叉加载（cross-loads）的词汇。如果一个词在多个维度中出现，根据其与各种子词的余弦相似度，将其归入最相似的单一维度，确保词典无重复。
- **`score_tf_idf(documents, document_ids, expanded_words, df_dict, N_doc, method)`**
  - **功能**: 核心打分函数。根据扩展词典统计词频，并支持通过 TF-IDF 或 WF-IDF (WF=log(1+TF)) 算法赋予权重，输出文档得分 DataFrame。

### 4.2 `culture.culture_models` 模块
- **`train_bigram_model(input_path, model_path)`**
  - **功能**: 使用 `gensim.models.phrases.Phrases` 从语料中学习共现频率高的词组（Bigram），并持久化模型。
- **`file_bigramer(input_path, output_path, model_path)`**
  - **功能**: 加载短语模型并转换语料，将句子中的常见词组通过下划线 `_` 连接，生成新的 N-gram 语料。
- **`train_w2v_model(input_path, model_path, ...)`**
  - **功能**: 基于 Trigram 级别的语料库，调用 `gensim.models.Word2Vec` 进行词向量训练。

### 4.3 `culture.preprocess` 模块
- **`class preprocessor(object)`**
  - **功能**: 封装了 `stanfordnlp.server.CoreNLPClient`，利用句法依赖树找出多词表达式（MWE），识别并替换命名实体（NER）。
- **`class text_cleaner(object)`**
  - **功能**: 提供文本清理链（Pipeline），主要通过正则与停用词表，去除数字、独立标点符号及被标记的无用 NER。

---

## 5. 依赖关系 (Dependencies)

项目依赖项定义在 `requirements.txt` 中。建议在 Python 3.8 - 3.12 环境下运行。

**核心依赖库:**
- `numpy >= 1.17.2` & `pandas >= 0.25.1`: 基础数据结构与表格处理。
- `gensim >= 3.7.2, < 4.0.0`: 核心 NLP 库，负责 Phrase 提取和 Word2Vec 训练。（注意：Gensim 版本限定在 4.0 以下，因为新版本重构了部分 API）。
- `scikit_learn >= 0.22.1`: 提供 L2 归一化等算法工具。
- `stanfordnlp == 0.2.0`: (可选) 用于调用 Stanford CoreNLP 进行深度中文句法分析。
- `jieba`: 用于快速的中文分词（配合 `jieba_tokenize.py` 使用）。
- `tqdm >= 4.31.1`: 提供命令行进度条。

---

## 6. 项目运行方式 (How to Run)

### 6.1 环境准备
1. 创建虚拟环境并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. （可选）如果使用 Stanford CoreNLP 解析，需要下载 `stanford-corenlp-4.5.4` 及对应的中文模型 jar 包，并在 `global_options.py` 中配置 `CORENLP_HOME` 环境变量。

### 6.2 数据准备
在 `data/input/` 目录下准备以下文件：
- `documents.txt`: 语料库，每行代表一篇完整文档。
- `document_ids.txt`: 文档 ID 列表，行数与 `documents.txt` 一一对应。
- `id2firms.csv`: (可选) 映射表，包含 `document_id`, `firm_id`, `time` 列，用于最后的分数聚合。

### 6.3 运行流水线
按照以下顺序依次执行脚本：

**步骤 1: 文本分词与解析**
```bash
python jieba_tokenize.py  # 推荐：快速 Jieba 分词
# 或者使用 python parse.py 进行 CoreNLP 深度解析
```
*(输出: `data/processed/parsed/documents.txt`)*

**步骤 2: 文本清洗与 Word2Vec 训练**
```bash
python clean_and_train.py
```
*(输出: 生成 bigram/trigram 语料，并保存 `models/w2v/w2v.mod` 等模型)*

**步骤 3: 种子词典扩展**
```bash
python create_dict.py
```
*(输出: `outputs/dict/expanded_dict.csv`，基于相似度计算出的文化词汇集)*

**步骤 4: 文档文化得分计算**
```bash
python score.py
```
*(输出: `outputs/scores/scores_TF.csv` 等，涵盖 TF/TFIDF/WFIDF 维度的得分)*

**步骤 5: 聚合至公司-年度层面 (可选)**
```bash
python aggregate_firms.py
```
*(输出: `outputs/scores/firm_scores_*.csv`)*
