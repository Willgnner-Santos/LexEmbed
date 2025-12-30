# Zero-Shot Legal Similarity Analysis

## 📋 Overview

This notebook (`A1.ipynb`) performs similarity analysis between legal cases using **embeddings from pre-trained language models**. The pipeline processes legal facts and thesis data, generates vector representations (embeddings), and applies clustering algorithms to group similar cases.

### Main Flow
```
Raw Data (CSV)
    ↓
Text Cleaning
    ↓
Data Preparation and Unification
    ↓
Embedding Generation (SentenceTransformer)
    ↓
Clustering and Evaluation
    ↓
Visualization with UMAP
```

---

## 🚀 How to Use

### Prerequisites
Install the necessary dependencies:

```bash
pip install pandas sentence-transformers scikit-learn hdbscan umap-learn matplotlib seaborn
```

### Basic Execution

1. **Place your CSV file** in the specified path or change the path in the code
2. **Execute the cells sequentially** in the notebook
3. **Analyze the results** of the clustering metrics

---

## 📝 Detailed Section Breakdown

### 1️⃣ Text Cleaning Functions (Cell 1)

**What it does:** Removes noise and standardizes the text of legal cases.

```python
# Remove punctuation at the beginning of text
remove_pontuacao_inicio(text)
# Remove repeated punctuation (more than 4 times)
remove_pontuacao_repetida(text)
```

**Examples:**
- `"!!!Important text"` → `"Important text"`
- `"Congratulations!!!!!!"` → `"Congratulations!!!!"` (keeps only 4 repetitions)

---

### 2️⃣ Data Loading (Cell 2)

**What it does:** Loads the CSV file with legal case data.

```python
df = pd.read_csv(r'/workspace/src/data/dados_berna_comparacao_fato_direito_04082025.csv', 
                  encoding='utf-8')
```

**Expected Columns:**
| Column | Description |
|--------|-----------|
| `fato_paradigma` | Description of the fact of the reference case |
| `fato_similar` | Description of the fact of the similar case |
| `direito_paradigma` | Legal thesis of the reference case |
| `direito_similar` | Legal thesis of the similar case |
| `processo_paradigma` | ID of the reference process |
| `processo_similar` | ID of the similar process |
| `similaridade_fato` | Similarity score of facts |
| `similaridade_tese` | Similarity score of theses |

---

### 3️⃣ Cleaning and Normalization (Cell 3)

**What it does:** Applies cleaning functions to all text columns.

```python
# Remove line breaks
df[cols] = df[cols].replace({r'[\r\n]+': ' '}, regex=True)

# Remove punctuation at the beginning
df[cols] = df[cols].map(remove_pontuacao_inicio)

# Remove extra spaces
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Remove repeated punctuation
df[cols] = df[cols].map(remove_pontuacao_repetida)
```

---

### 4️⃣ Data Selection and Reduction (Cell 4)

**What it does:** Selects up to 5 similar cases per paradigm fact and limits to 500 rows.

```python
df_5_fatos = (
    df.drop_duplicates(subset=['fato_paradigma', 'fato_similar'])
      .groupby('fato_paradigma')
      .head(5)  # ← Maximum 5 similar cases per fact
      .reset_index(drop=True)
)
df = df_5_fatos.head(500)  # ← Limits to 500 rows
```

**Why?** Reduces data volume for faster processing and avoids imbalance.

---

### 5️⃣ Duplicate Removal (Cell 5)

**What it does:** Removes duplicate records to ensure unique data.

```python
df_unique = df.drop_duplicates(subset=['fato_similar', 'direito_similar'])
print(f"Total unique rows: {len(df_unique)}")
```

---

### 6️⃣ Data Unification and Normalization (Cell 6)

**What it does:** Combines facts and theses and normalizes similarity scores.

```python
# Merges fact + thesis for each case
df["paradigma_unificado"] = df["fato_paradigma"] + " " + df["direito_paradigma"]
df["similar_unificado"] = df["fato_similar"] + " " + df["direito_similar"]

# Calculates the average of similarities and normalizes
df["similaridade_normalizada"] = (df["similaridade_fato"] + df["similaridade_tese"]) / 2

# Creates final DataFrame for embeddings
df = pd.DataFrame({
    "processo_paradigma": df["processo_paradigma"],
    "processo_similar": df["processo_similar"],
    "sentence1": df["paradigma_unificado"],
    "sentence2": df["similar_unificado"],
    "score": df["similaridade_normalizada"]
})
```

**Result:** Each row contains a pair of texts (paradigm vs similar) with a similarity score.

---

### 7️⃣ Text Extraction (Cell 7)

**What it does:** Extracts similar texts into a list for processing.

```python
sentences = df['sentence2'].tolist()
```

---

### 8️⃣ Embedding Generation (Cell 8) ⭐

**What it does:** Converts texts into numerical vectors (embeddings) using a language model.

```python
from sentence_transformers import SentenceTransformer

# Loads pre-trained model
model = SentenceTransformer(r'/workspace/Similaridade-Fato-Tese/modelo_ft_similaridade',
                            trust_remote_code=True)

# Generates embeddings
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=16)
```

**What are embeddings?** They are vector representations of texts that capture semantic meaning. Similar texts have vectors close together in dimensional space.

---

### 9️⃣ Clustering and Evaluation (Cell 9)

**What it does:** Groups similar texts and evaluates clustering quality.

#### Dimensionality Reduction with UMAP
```python
# Reduces embeddings from high dimension to 2D for visualization
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)
```

#### Tested Clustering Algorithms
| Algorithm | Parameters | Usage |
|-----------|-----------|-----|
| **KMeans** | `n_clusters=300` | Centroid-based clustering |
| **AgglomerativeClustering** | `n_clusters=300` | Hierarchical clustering |
| **SpectralClustering** | `n_clusters=300` | Graph-based clustering |
| **AffinityPropagation** | `preference=-5.0` | Automatically finds exemplars |
| **DBSCAN** | `eps=0.025` | Density-based clustering |
| **HDBSCAN** | (no params) | Hierarchical DBSCAN (best for non-uniform data) |

#### Metric: Silhouette Score
```python
score = silhouette_score(data, labels, metric=metric)
```

**What does it mean?** 
- Range: `-1` to `+1`
- `Close to +1` = Well-defined clusters ✅
- `Close to 0` = Overlapping clusters ⚠️
- `Negative` = Poor classifications ❌

---

## 🔄 How to Change Models

### Option 1: Use Pre-trained Model from HuggingFace

```python
from sentence_transformers import SentenceTransformer

# Replace this line (Cell 8)
model = SentenceTransformer('distiluse-base-multilingual-case-v2')  # For multilingual texts
# or
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# or for Portuguese-specific
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

embeddings = model.encode(sentences, show_progress_bar=True, batch_size=16)
```

### Option 2: Use Custom Local Model

If you have a fine-tuned model saved locally:

```python
model = SentenceTransformer(r'./path/to/your/model', trust_remote_code=True)
```

### Option 3: Use Another Embedding Type (OpenAI, for example)

```python
from openai import OpenAI

client = OpenAI(api_key="your_api_key")

def get_embeddings_openai(sentences):
    embeddings = []
    for sentence in sentences:
        response = client.embeddings.create(
            input=sentence,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

embeddings = get_embeddings_openai(sentences)
```

### Recommended Models by Use Case

| Use Case | Model | Advantages |
|----------|-------|-----------|
| **Portuguese Legal** | `neuralmind/bert-base-portuguese-cased` | Trained for Portuguese |
| **Multilingual** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Works in 50+ languages |
| **Fast and Lightweight** | `all-MiniLM-L6-v2` | Small size, good performance |
| **More Accurate** | `all-mpnet-base-v2` | Better quality, slower |
| **Custom** | Your fine-tuned model | Optimized for your domain |

---

## 📊 Interpreting Results

### Example Output

```
KMeans: Silhouette Score = 0.4521 (clusters: 300)
AgglomerativeClustering: Silhouette Score = 0.3892 (clusters: 300)
HDBSCAN: Less than 2 clusters found. Silhouette Score not calculated.
```

**What does it mean?**
- KMeans performed better with score 0.45
- HDBSCAN did not find significant clusters (data too concentrated)
- Consider adjusting parameters if scores are low

---

## 🎯 Common Customizations

### Increase/Decrease Number of Clusters

```python
# Cell 9: change 300 to another value
(cluster.KMeans, {'n_clusters': 100, 'random_state': 42}),  # Fewer clusters
```

### Increase Data Volume

```python
# Cell 4: remove the limitation
df = df_5_fatos  # Process all data
```

### Use Different Metric for DBSCAN

```python
# Cell 9: change eps and metric according to dimensionality
(cluster.DBSCAN, {'eps': 0.05, 'metric': 'euclidean'}),  # Larger eps = larger clusters
```

---

## 🐛 Troubleshooting

### Error: "Model not found"
```
Solution: Check the model path or download via:
model = SentenceTransformer('model-name')  # Automatic download
```

### UMAP Slow
```
Solution: Reduce n_neighbors or use metric='cosine' (already used)
```

### Negative Silhouette Scores
```
Solution:
1. Change the number of clusters
2. Use HDBSCAN algorithm (detects automatically)
3. Check input data quality
```

---

## 📚 References

- [Sentence Transformers](https://www.sbert.net/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [UMAP](https://umap-learn.readthedocs.io/)

---

## 📝 Important Notes

✅ **What this notebook does well:**
- Processes large volumes of legal texts
- Generates robust semantic embeddings
- Evaluates clustering quality with multiple metrics

⚠️ **Limitations:**
- Requires GPU for very large datasets (>100k texts)
- Quality depends on the chosen embedding model
- Unsupervised clustering requires manual parameter tuning

---

**Last updated:** December 2025
