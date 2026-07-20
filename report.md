# Self-Explanatory Disinformation Detection System with Human in the Loop

## 1. Algorithm Overview

Key features of the proposed system:

- **Interpretable predictions** — every classification is accompanied by the matched narrative and its position in the hierarchy, explaining *why* a text was flagged
- **Incremental updates** — new data can be ingested into the existing trees without a full retrain, keeping the knowledge base up to date
- **Human-in-the-loop control** — domain experts can directly edit, remove, or restructure narrative nodes through the web UI, correcting the system's behavior in real time
- **Reproducible pipeline** — all training, evaluation, and dataset commands are available via `make` targets (run `make help` to list them)

### 1.1 Narrative Tree Construction

The algorithm builds two separate hierarchical narrative trees, one from known disinformation 
articles and one from trustable news publications. Each text is first encoded into a dense vector using
SBERT (`all-MiniLM-L6-v2` - en, `paraphrase-multilingual-MiniLM-L12-v2`-ro). 
The tree is constructed through iterative agglomerative clustering: 
starting at a cosine distance threshold of 0.1 and incrementing by 0.05 up to 0.8, texts are grouped into 
clusters at each level. 
For each cluster, a large language model (Gemma 3) generates a single-sentence narrative summary 
that captures the common narrative of the grouped texts. 
This narrative becomes the parent node of the cluster, with the original texts 
(or previous-level narratives) as its children.
The process repeats at increasing distance thresholds, progressively merging finer clusters into broader
narrative themes, until the tree reaches a stable hierarchical structure.

### 1.2 Dual-Tree Classification of statements

To classify a statement/sentence, the system retrieves the 
closest matching nodes from both the fake and real narrative trees. 
Retrieval works in two stages: first, cosine similarity selects the top-10 candidate nodes from each tree; 
then, a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) scores each candidate against the input text. 
The classification decision follows a four-case logic:

1. **Both scores very low** (< -5): no match in either tree (the score signals irrelevance to the known information), default to REAL because information is unknown
2. **Strong clear winner** (score > 0 and score difference > 3.0): If one tree scores significantly higher than the other, the classification is decided directly: the label of the higher-scoring tree is returned without calling the LLM, since the knowledge base already provides a decisive match.
3. **Ambiguous cases**: When the knowledge base contains relevant narratives on both the true and fake sides, the system constructs an argument using matching narratives from each tree. A judge LLM then evaluates whether the disinformation narratives demonstrate adherence of the input text to known fake narratives. This is the most critical branch of the algorithm — it leverages the hierarchical tree structure to provide context and arguments from both sides, enabling a well-informed classification even when the scores alone are inconclusive.

This tiered approach ensures that the expensive LLM call is only made for ambiguous cases giving historical context and overview on existing narratives, while clear-cut decisions are resolved through score comparison alone.

### 1.3 Classification of Complete News Articles

To classify a full-length news article, the system applies the dual-tree classifier at sentence granularity and aggregates the results. The article summary is split into individual sentences using NLTK's `sent_tokenize`. The title and each sentence are independently classified through the dual-tree pipeline described in Section 1.2. Each sentence that is classified as fake is paired with the narrative node it matched, building an interpretable audit trail.

The article-level label is determined by an adaptive threshold that scales with article length:

- **1–3 sentences:** 1 fake sentence is sufficient to classify the article as fake.
- **4–5 sentences:** At least 2 fake sentences are required.
- **6+ sentences:** A sliding window of size 6 scans the sentence sequence; if any window contains ≥3 fake sentences, the article is classified as fake. Additionally, if any window contains ≥2 fake sentences and the title was also classified as fake, the article is labeled fake.

This adaptive approach prevents short articles from being misclassified due to overly strict thresholds, while longer articles require sustained disinformation signal across multiple sentences.

### 1.4 Incremental Data Ingestion

The narrative trees support incremental updates to the KB, allowing new information to be incorporated without rebuilding the entire tree from scratch. When new labeled data becomes available, the ingestion process works as follows:

1. **Best-node matching:** Each new article is embedded with SBERT and matched against the existing tree (depending if the article is true or fake) using the same two-stage retrieval (cosine similarity + cross-encoder reranking). The best-matching leaf node is identified.
2. **Grafting:** The new article is attached as a child of the matched leaf node, extending the tree at the point of highest semantic relevance.
3. **Local re-clustering:** Only the modified subtrees are re-clustered using `cluster_new_items()`, which applies agglomerative clustering locally to merge the newly grafted items with their siblings. This avoids the cost of a full tree rebuild.
4. **Narrative regeneration:** The LLM regenerates narrative summaries only for nodes whose children changed, propagating upward through the affected branch. Unmodified subtrees retain their existing narratives.

This incremental approach reduces update since only a small fraction of the tree is modified. 

### 1.5 Transparency and Human-in-the-Loop

Unlike black-box classifiers, the system provides a full explanation for every prediction. When a text is classified as fake, the output includes the specific narrative it matched and that narrative's position within the hierarchical tree. For complete news articles, each sentence that triggered a fake classification is paired with its matched narrative, producing a transparent audit trail.

**Example — classifying the statement "Key posts are being filled by Russophobic Balts.":**

The dual-tree pipeline retrieves the closest nodes from both trees and returns:
- **Label:** FAKE
- **Matched narrative:** *Baltic states are aggressively hostile towards Russia.*
- **Narrative parent in the tree:** *Anti-Russian sentiment in Eastern Europe*

The human operator can immediately see *why* the system flagged this sentence and verify whether the matched narrative is appropriate. If the match is incorrect — for example, if the narrative tree contains a misclassified node or an overly broad cluster — the operator can directly modify the tree structure:

1. **Correct a narrative label:** Delete a node from the fake tree if it was incorrectly categorized.
2. **Split or merge clusters:** If a narrative cluster groups unrelated topics, the operator can split it
3. **Edit narrative summaries:** Refine LLM-generated narrative descriptions that are too vague or inaccurate.
4. **Remove nodes:** Delete irrelevant or noisy nodes that cause false positives.

These corrections propagate through subsequent classifications immediately, since the system reads the tree structure at inference time. This human-in-the-loop design ensures that domain experts retain full control over the knowledge base, while the automated pipeline handles the scale of classification.

## 2. Datasets 

This section describes the datasets used to build and evaluate the system.

### 2.1 MindBugs Dataset — the dataset behind the system
The MindBugs dataset serves as the foundational knowledge base of the system — its labeled articles are used to build the narrative trees that power classification.
#### Fake news sources
* **EUvsDisinfo:** 17,226 articles (also used for full-text evaluation)
* **Veridica:** 1,038 articles
* **Factual.ro:** 334 articles

These include political disinformation articles from Romania and international fact-checking sources.

#### Real news sources
* **Zetta Cloud:** 18,598 verified articles

The collection includes news from various sites such as dailymail.co.uk, straitstimes.com, foxnews.com and others. These were ranked according to the "TrustScore" defined by Zetta.

### 2.2 Evaluation Datasets
In order to evaluate the system, multiple datasets were used to test its performance across different domains, languages, and disinformation styles.

| Dataset | Domain                                     | Train | Val | Test | Labels |
|---------|--------------------------------------------|------:|----:|-----:|--------|
| Mindbugs | EU disinformation + factual statements     | 22,820 | 2,934 | 3,912 | fake / real |
| COVID-19 | COVID-related claims                       | ~2,100 | ~200 | ~200 | fake / real |
| LIAR | Political statements (6-class, binarized)  | ~10,000 | ~1,200 | ~1,200 | fake / real |
| FakeNewsNet | PolitiFact + GossipCop articles            | ~2,300 | ~290 | ~290 | fake / real |

### 2.3 Complete News Articles Dataset

For evaluating the system on full-length news articles (as opposed to short claims or summaries), a separate dataset of 600 complete articles was compiled from the Mindbugs sources. The dataset contains 300 fake articles sourced from EUvsDisinfo and 300 real articles, each including a title and full-text body. This dataset is used exclusively for testing the sentence-level classification pipeline described in Section 1.3.

## 3. Evaluation Results

Each dataset is used independently: the system is trained and evaluated on the same dataset using a train/test split. This measures how well the narrative tree captures the patterns within each domain. The table below shows the best result per dataset, selected by optimal clustering threshold. Validation is reported on a capped 800-sample subset per dataset:

| Dataset | Threshold |  Accuracy  | Precision | Recall | F1 |
|---------|:---------:|:----------:|:---------:|:------:|:--:|
| Mindbugs |   0.50    | **0.9356** |  0.9356   | 0.9356 | 0.9356 |
| COVID-19 |   0.60    | **0.8650**  |   0.8729  | 0.8650 | 0.8636 |
| LIAR |   0.50    | **0.5988** |  0.5999   | 0.5988 | 0.5987 |
| FakeNewsNet |   0.65    | **0.8013** |  0.8003   | 0.8013 | 0.8008 |

The algorithm performs best on Mindbugs (93.6%) and COVID-19 (86.5%), where disinformation follows recognizable recurring narratives. Performance is lower on LIAR (59.9%), which contains fine-grained political statements that are harder to group into narrative clusters.

On the held-out **Mindbugs test split** (3,826 statements, not used for tuning), the classifier reaches **0.9375** accuracy and F1 at threshold 0.50, essentially matching the validation figure (0.9356) and confirming that the narrative tree generalizes rather than overfitting the tuning split.

Confusion matrix (Mindbugs test, threshold 0.50):

|  | Predicted Real | Predicted Fake |
|--|:--------------:|:--------------:|
| **Actual Real** |      1792      |      121       |
| **Actual Fake** |      118       |      1795      |

## 4. Comparison with Baseline Algorithms

Validation accuracy and F1 across all datasets. Baselines use TF-IDF features.

| Algorithm | Mindbugs  |  | COVID-19 |  | LIAR |  | FakeNewsNet |  | **Average** |  |
|-----------|:---------:|:--:|:--------:|:--:|:----:|:--:|:-----------:|:--:|:------------:|:--:|
|  |    Acc    | F1 | Acc | F1 | Acc | F1 | Acc | F1 |     Acc      | F1 |
| **Narrative Tree (ours)** | 0.936 | 0.936 | 0.865 | 0.864 | 0.599 | 0.599 | 0.801 | 0.801 | **0.800** | **0.800** |
| SVM | 0.952 | 0.952 | 0.935 | 0.935 | 0.576 | 0.574 | 0.836 | 0.829 | **0.825** | **0.822** |
| Logistic Regression | 0.951 | 0.951 | 0.928 | 0.928 | 0.604 | 0.596 | 0.836 | 0.816 | **0.829** | **0.823** |
| Gradient Boosting | 0.879 | 0.879 | 0.868 | 0.868 | 0.583 | 0.533 | 0.809 | 0.768 | **0.785** | **0.762** |
| Decision Tree | 0.899 | 0.899 | 0.856 | 0.855 | 0.550 | 0.549 | 0.784 | 0.781 | **0.772** | **0.771** |
| KNN | 0.905 | 0.905 | 0.913 | 0.913 | 0.569 | 0.567 | 0.803 | 0.784 | **0.798** | **0.792** |

| Property | Narrative Tree (ours) | SVM | LR | GB | DT | KNN |
|----------|:----:|:---:|:--:|:--:|:--:|:---:|
| Interpretable predictions | Yes | No | No | No | Yes | Partial |
| Incremental update (no retrain) | Yes | No | No | No | No | Yes |
| Human-in-the-loop editing | Yes | No | No | No | No | No |
| Narrative-level explanation | Yes | No | No | No | No | No |

The narrative tree algorithm is competitive with traditional ML baselines, performing within roughly two points of the strongest models on Mindbugs (0.936 vs 0.952 for SVM) and FakeNewsNet, and close to the best on LIAR (0.599 vs 0.604 for Logistic Regression). SVM and Logistic Regression achieve higher average accuracy, but these are black-box classifiers that require full retraining on new data. In contrast, the narrative tree provides an interpretable explanation (the matched narrative and its position in the hierarchy) alongside every prediction, supports incremental updates through the ingestion pipeline without rebuilding the model from scratch, and allows domain experts to directly edit, remove, or restructure narrative nodes — giving humans the ability to correct the system's behavior in real time.

## 5. Clustering Threshold Sensitivity — Mindbugs

Performance of the dual-tree classifier across different clustering distance thresholds used during tree construction.

At threshold 0.0, no clustering is performed: the system operates as a flat database with simple semantic search and an LLM judge, with no synthetic narrative nodes added.

As the threshold increases, articles are grouped into progressively broader clusters and the LLM generates narrative summaries for each cluster, adding hierarchical context and information to the tree.
The threshold was incremented by 0.05 starting from 0.0, and tree construction stopped once no improvement was observed for three consecutive iterations:

### 5.1 Gemma 27B

| Threshold | Val F1 | Val Accuracy | Val Precision | Val Recall |
|:---------:|:------:|:------------:|:-------------:|:----------:|
| 0.00 | 0.9163 | 0.9163 | 0.9163 | 0.9163 |
| 0.10 | 0.9158 | 0.9158 | 0.9158 | 0.9158 |
| 0.15 | 0.9153 | 0.9153 | 0.9153 | 0.9153 |
| 0.20 | 0.9163 | 0.9163 | 0.9163 | 0.9163 |
| 0.25 | 0.9155 | 0.9155 | 0.9156 | 0.9155 |
| 0.30 | 0.9135 | 0.9135 | 0.9135 | 0.9135 |
| 0.35 | 0.9155 | 0.9155 | 0.9155 | 0.9155 |
| 0.40 | 0.9176 | 0.9176 | 0.9179 | 0.9176 |
| 0.45 | 0.9217 | 0.9217 | 0.9217 | 0.9217 |
| **0.50** | **0.9356** | **0.9356** | **0.9356** | **0.9356** |
| 0.55 | 0.9204 | 0.9204 | 0.9205 | 0.9204 |
| 0.60 | 0.9181 | 0.9181 | 0.9182 | 0.9181 |
| 0.65 | 0.9191 | 0.9191 | 0.9194 | 0.9191 |
| 0.70 | 0.9196 | 0.9196 | 0.9200 | 0.9196 |
| 0.75 | 0.9199 | 0.9199 | 0.9201 | 0.9199 |
| 0.80 | 0.9196 | 0.9196 | 0.9198 | 0.9196 |

Best threshold on validation: **0.50** (Accuracy=0.9356, F1=0.9356)

Validation F1 ranges from 91.3% to 93.6%, a span of approximately 2.2 percentage points across the full threshold range. Performance is relatively stable in the low-90s across most thresholds, peaking at 0.50 where narrative clusters reach an effective level of abstraction. The 0.50 setting stands out as the clear best; the surrounding thresholds sit around 91-92%.

## 9. Complete News Article Evaluation

Previous evaluations (Sections 3–8) classify short texts — individual claims or article summaries. 
To test the system on full-length news articles, we apply a sentence-level decomposition strategy. 
The methodology is explained at Section 1.3.

### 9.1 Narrative Attribution

A key advantage of the sentence-level approach is that the system provides an interpretable explanation: each fake sentence is paired with the narrative it matched in the fake tree. These pairs are aggregated into a ranked list of narratives with their occurrence count.

**Example — "The new European Commission is all geared up for war with Russia":**

Sentence–narrative pairs:

| Sentence                                                                                                                                                                           | Matched Narrative |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| The new European Commission is all geared up for war with Russia                                                                                                                   | The EU is preparing for a new war against Russia. |
| The commission is all prepared to invade Russia.                                                                                                                                   | The EU is preparing for a new war against Russia. |
| Key posts are being filled by Russophobic Balts.                                                                                                                                   | Pulitzer prize to The New York Times is Russophobic |
| The orientation is clear: against Russia.                                                                                                                                          | The West is provoking Russia |
| The three Baltic states are indeed among the declared champions of aggressiveness towards Moscow.                                                                                  | Baltic states are aggressively hostile towards Russia. |
| This goes so far that the position could also have been called 'Commissioner for the War against Russian', a prospect that, according to Vilnius, absolutely must be prepared for. | The EU is openly preparing a war against Russia |

Aggregated narrative counts:

| Narrative | Count |
|-----------|:-----:|
| The EU is preparing for a new war against Russia. | 2 |
| Pulitzer prize to The New York Times is Russophobic | 1 |
| The West is provoking Russia | 1 |
| Baltic states are aggressively hostile towards Russia. | 1 |
| The EU is openly preparing a war against Russia | 1 |

The system correctly identifies the dominant disinformation theme (EU preparing war against Russia) and attributes specific sentences to their closest known narratives, providing a transparent audit trail for each prediction.

### 9.2 Results

| Metric | Score  |
|--------|:------:|
| Accuracy | 0.9700 |
| Precision | 0.9700 |
| Recall | 0.9700 |
| F1-score | 0.9700 |

Confusion matrix:

|  | Predicted Real | Predicted Fake |
|--|:--------------:|:--------------:|
| **Actual Real** |      291       |       9        |
| **Actual Fake** |       9        |      291       |

The system achieves 97.0% accuracy on complete news articles, outperforming the short-text evaluation (93.6% on Mindbugs at threshold 0.50). This improvement is expected: full-length articles contain multiple sentences that reinforce the disinformation signal, making the sliding-window aggregation more robust than single-sentence classification. Only 9 real articles were misclassified as fake (false positives) and 9 fake articles were missed (false negatives).

## 9. In-the-Wild Experiment: TVR Info

To evaluate the system's behavior on real-world, unlabeled news content,
we applied the complete news article classification pipeline (Section 1.3) to 150 articles scraped from [https://tvrinfo.ro/](https://tvrinfo.ro/), the online platform of the Romanian public television broadcaster. Since TVR Info is a mainstream, state-affiliated news outlet, the expectation is that the vast majority of articles should be classified as real. Any articles flagged as fake represent either genuine disinformation overlap or false positives, providing insight into the system's specificity on in-the-wild data.

### Results

| Metric | Value |
|--------|:-----:|
| Total articles analyzed | 150 |
| Flagged as fake | 17 |
| Flagged as real | 133 |
| Fake rate (%) | 11.3% |

### Articles Flagged as Fake

Of the 17 articles flagged as fake, most are incidental matches — sports, crime, and domestic-politics stories that brushed against an unrelated narrative. The clearest cases, where the article's content directly echoes a known disinformation narrative, are listed below with their single dominant match. In each, the system correctly detects narrative *overlap* but cannot distinguish factual reporting *about* a sensitive topic from disinformation *promoting* it.

| # | Article Title | Main Matched Narrative |
|:-:|---------------|------------------------|
| 1 | [Iran: EU countries will "pay the price" if they don't speak out on US-Israeli attacks](https://tvrinfo.ro/iran-tarile-ue-vor-plati-pretul-mai-devreme-sau-mai-tarziu-daca-nu-se-exprima-in-legatura-cu-atacurile-americano-israeliene/) | European countries are under heavy US control |
| 2 | [Washington undecided on the duration of the offensive against Iran](https://tvrinfo.ro/washingtonul-indecis-asupra-duratei-ofensivei-impotriva-iranului-intre-opt-saptamani-si-septembrie/) | US plans to invade Iran |
| 3 | [US military campaign in Iran "will last, but won't be like Iraq"](https://tvrinfo.ro/campania-militara-americana-in-iran-va-dura-dar-nu-va-fi-la-fel-ca-in-irak-potrivit-secretarului-american-al-apararii/) | Trump's main goal is to destroy the current regime in Iran |
| 4 | ["Broad support" in Europe for strikes against Iran, per NATO's secretary general](https://tvrinfo.ro/sprijin-larg-in-europa-pentru-loviturile-contra-iranului-potrivit-secretarului-general-al-nato/) | NATO expansion is causing instability in Europe |
| 5 | [Polish FM: Warsaw remains a loyal US ally, "but we can't be suckers"](https://tvrinfo.ro/ministrul-polonez-de-externe-varsovia-a-fost-si-va-ramane-un-aliat-loial-al-americii-dar-nu-putem-fi-fraieri/) | The United States controls Poland's government |
| 6 | [Bill and Hillary Clinton testify behind closed doors in the Epstein case](https://tvrinfo.ro/bill-clinton-si-sotia-sa-hillary-depune-marturie-cu-usile-inchise-in-dosarul-epstein/) | The West refuses to investigate crimes involving global elites |
| 7 | [Explosive revelations: Bill Gates admits connection with Epstein](https://tvrinfo.ro/dezvaluiri-explozive-bill-gates-recunoaste-legatura-cu-epstein-si-ca-si-a-inselat-sotia-cu-doua-rusoaice/) | Epstein's files reveal connections with the heart of the EU |
| 8 | [Four occupants of a US-registered ship killed by Cuban border guards](https://tvrinfo.ro/patru-ocupanti-ai-unei-nave-inmatriculate-in-sua-ucisi-de-graniceri-cubanezi-intr-o-confruntare-in-apele-teritoriale-ale-insulei/) | The United States is actively undermining the Cuban government |
