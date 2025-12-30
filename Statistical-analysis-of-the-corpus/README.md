# Statistical Analysis of the Corpus

## 📋 Overview

This notebook (`Statistical-analysis-of-the-corpus.ipynb`) performs comprehensive **statistical and linguistic analysis of a legal corpus**, including data consolidation, text cleaning, lexical metrics computation, tokenization analysis, and embedding-based visualizations. It provides insights into text characteristics suitable for NLP model training and evaluation.

### Main Flow
```
Load CSV Data
    ↓
Merge & Export to JSON
    ↓
Remove Stopwords & Clean Text
    ↓
Compute Textual Metrics (words, chars, sentences, diversity)
    ↓
Analyze BERT Tokenization
    ↓
Generate Embeddings & Visualize (UMAP, t-SNE)
```

---

## 🚀 How to Use

### Prerequisites

Install required dependencies:

```bash
pip install pandas nltk transformers sentence-transformers scikit-learn umap-learn matplotlib numpy
```

### Basic Execution

1. **Prepare CSV files** - Ensure they have the required text columns
2. **Update file paths** - Modify paths in cells to match your data location
3. **Run cells sequentially** - Execute from top to bottom
4. **Review outputs** - Check metrics tables and visualizations

### Expected Outputs

- Cleaned JSON files (with and without stopwords)
- Statistical tables (textual metrics)
- Histograms and bar charts
- UMAP and t-SNE visualizations
- BERT tokenization analysis

---

## 📝 Detailed Section Breakdown

### 1️⃣ Data Loading and Merging (Cell 1)

**What it does:** Loads multiple CSV files and consolidates them into a single JSON dataset.

```python
import pandas as pd
import json
from pathlib import Path

# Input file paths (two CSV exports from different dates)
csv_1 = "/content/dados_berna_comparacao_fato_direito_04082025.csv"
csv_2 = "/content/dados_berna_comparacao_fato_direito_24072025.csv"

# Load both CSV files
df1 = pd.read_csv(csv_1)
df2 = pd.read_csv(csv_2)

# Concatenate into single DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# Optional: Remove exact duplicates (commented by default)
# df = df.drop_duplicates()

# Display first 5 records
exemplos_5 = df.head(5).to_dict(orient="records")
print(json.dumps(exemplos_5, ensure_ascii=False, indent=2))

# Print total records
print(f"Total records: {len(df)}")

# Export to unified JSON
out_json = "/content/dados_berna_comparacao_fato_direito_unificado.json"
df.to_json(out_json, orient="records", force_ascii=False, indent=2)
print(f"JSON saved at: {out_json}")
```

**Key Operations:**

| Operation | Purpose | Note |
|-----------|---------|------|
| `pd.concat()` | Merge DataFrames | Preserves all records |
| `drop_duplicates()` | Remove exact dupes | Optional, currently disabled |
| `to_json()` | Export to JSON | Preserves Unicode (Portuguese) |

**Expected Text Fields:**
- `fato_paradigma` - Facts of reference case
- `fato_similar` - Facts of similar case
- `direito_paradigma` - Legal thesis of reference
- `direito_similar` - Legal thesis of similar case

---

### 2️⃣ Stopword Removal and Text Cleaning (Cell 2-3)

**What it does:** Removes Portuguese stopwords and cleaning artifacts from text.

#### Setup and Dependencies (Cell 2)

```python
!pip -q install nltk

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS_PT = set(stopwords.words("portuguese"))

print(f"Portuguese stopwords count: {len(STOPWORDS_PT)}")
```

**NLTK Portuguese Stopwords (sample):**
```
'a', 'o', 'é', 'com', 'de', 'para', 'em', 'um', 'uma', 
'que', 'se', 'não', 'do', 'dos', 'da', 'das', ...
```

#### Advanced Stopword Removal (Cell 3)

**What it does:** Intelligently removes stopwords while preserving punctuation and structure.

```python
EXTRA_REMOVE_TOKENS = {
    "\u00a0",  # Non-breaking space
    """, """, "'", "'", "´", "`",  # Smart quotes
}

# Pattern preserves punctuation as separate tokens
_token_pat = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def remove_stopwords_keep_punct(text, stopwords_set, extra_remove):
    """
    Remove Portuguese stopwords but keep punctuation intact.
    Returns (cleaned_text, removed_tokens).
    """
    if not isinstance(text, str) or not text.strip():
        return text, []

    tokens = _token_pat.findall(text)
    removed = []
    kept = []

    for token in tokens:
        token_lower = token.lower()

        # Remove explicit junk tokens
        if token in extra_remove:
            removed.append(token)
            continue

        # Remove stopwords only for word tokens (not punctuation)
        if re.search(r"\w", token) and token_lower in stopwords_set:
            removed.append(token)
            continue

        kept.append(token)

    # Reconstruct text and normalize spacing
    output = " ".join(kept)
    output = _fix_punct_spacing(output)
    return output, removed
```

**Removal Policy:**

✅ **Removed:**
- Standard PT-BR stopwords (NLTK list: ~250 words)
- Extra junk tokens (smart quotes, NBSP, etc.)
- Word tokens only (not punctuation)

✅ **Preserved:**
- Numbers and digits
- Punctuation marks (as separate tokens)
- Sentence structure

**Example Transformations:**

```
Before: "O texto é muito importante para a análise!!!!!!!"
After:  "texto muito importante análise!!!!"

Before: "Se não houver recurso, será encerrado."
After:  "recurso, será encerrado."
```

**Statistics Generated:**

```python
# Top 30 removed stopwords (global)
print(removed_counter_all.most_common(30))
# Output: [('de', 1245), ('da', 892), ('que', 756), ...]

# Top 15 removed per field
for field in TEXT_FIELDS:
    print(f"\n[{field}]")
    print(removed_counter_by_field[field].most_common(15))
```

---

### 3️⃣ Basic Textual Metrics (Cell 4)

**What it does:** Computes fundamental linguistic statistics per text field.

#### Tokenization and Splitting

```python
def sent_split_simple(txt):
    """Split text into sentences using period/exclamation/question marks."""
    return [s for s in re.split(r'(?<=[\.\!\?])\s+', txt) if s.strip()]

def word_tokenize_simple(txt):
    """Split text into words and punctuation (keeps both as tokens)."""
    return re.findall(r"\w+|\S", txt, re.UNICODE)
```

#### Lexical Diversity Measures

**Type-Token Ratio (TTR):**
```python
def lexical_diversity(tokens):
    """
    TTR = vocabulary size / token count
    Range: 0-1 (higher = more diverse vocabulary)
    """
    if not tokens:
        return 0.0
    types = len(set([t.lower() for t in tokens if re.search(r"\w", t)]))
    return types / len(tokens)
```

**Interpretation:**
- **TTR > 0.7** = Very diverse vocabulary (more unique words)
- **TTR 0.5-0.7** = Moderately diverse
- **TTR < 0.3** = Limited vocabulary (more repetition)

**Herdan's C Index:**
```python
def herdan_c(tokens):
    """
    C = log(vocabulary size) / log(token count)
    More stable than TTR (doesn't decrease with text length as much)
    Range: 0-1
    """
    vocab_size = len(set([t.lower() for t in tokens if re.search(r"\w", t)]))
    token_count = len([t for t in tokens if re.search(r"\w", t)])
    
    if vocab_size < 2 or token_count < 2:
        return 0.0
    return math.log(vocab_size) / math.log(token_count)
```

**Interpretation:**
- **C > 0.5** = Good lexical diversity
- **C 0.3-0.5** = Moderate diversity
- **C < 0.3** = Limited diversity

#### Metrics Computed

```python
stats = [
    {
        "Campo Textual": "field_name",  # Text field
        "Avg. Words": mean_words,       # Average word count
        "Std. Words": std_words,        # Standard deviation
        "Avg. Chars": mean_chars,       # Average character count
        "Std. Chars": std_chars,        # Standard deviation
        "Avg. Sents": mean_sents,       # Average sentence count
        "LexDiv Avg. (TTR)": mean_ttr,  # Average type-token ratio
        "Herdan C Avg.": mean_c,        # Average Herdan's C
        "N": count                      # Sample size
    }
]
```

#### Visualizations

```python
# Bar chart: Average word length per field
bar_with_error(stats_df, "Avg. Words", "Average word count", "Words")

# Bar chart: Average character length
bar_with_error(stats_df, "Avg. Chars", "Average character count", "Characters")

# Bar chart: Sentence count
bar_with_error(stats_df, "Avg. Sents", "Average sentence count", "Sentences")

# Bar chart: Lexical diversity (TTR)
plt.bar(range(len(stats_df)), stats_df["LexDiv Avg. (TTR)"])

# Bar chart: Lexical diversity (Herdan C)
plt.bar(range(len(stats_df)), stats_df["Herdan C Avg."])
```

**Key Insights:**
- Longer texts (more words/chars) = more complex documents
- Higher TTR/Herdan C = greater vocabulary richness
- High standard deviation = heterogeneous text lengths

---

### 4️⃣ BERT Tokenization Analysis (Cell 5)

**What it does:** Analyzes text length in subword tokens (WordPiece) and identifies texts exceeding BERT's 512-token limit.

```python
from transformers import AutoTokenizer

TOKENIZER_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def count_wordpiece_tokens(texts):
    """Count WordPiece tokens for each text."""
    token_lengths = []
    for text in texts:
        # encode includes special tokens ([CLS], [SEP])
        token_ids = tokenizer.encode(str(text), add_special_tokens=True, truncation=False)
        token_lengths.append(len(token_ids))
    return np.array(token_lengths)

# Compute token lengths per field
bert_len = {}
for field in TEXT_FIELDS:
    lengths = count_wordpiece_tokens(df_clean[field].astype(str).tolist())
    bert_len[field] = lengths
```

**Why This Matters:**

- BERT maximum sequence length: **512 tokens**
- Texts > 512 tokens must be truncated (information loss)
- Portuguese is more verbose than English (~20% more tokens)

#### Histogram Visualization

```python
# Per-field token length distribution
for field in TEXT_FIELDS:
    plt.hist(bert_len[field], bins=50)
    plt.title(f"Token Length Distribution — {field}")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.show()
```

**Interpretation:**
- **Sharp peak at 512** = Many texts truncated
- **Right tail** = Texts exceeding limit
- **Uniform distribution** = Diverse text lengths

#### Over-512 Analysis

```python
# Percentage of texts exceeding 512 tokens
over_512_pct = []
for field in TEXT_FIELDS:
    lengths = bert_len[field]
    pct = 100.0 * (lengths > 512).sum() / len(lengths)
    over_512_pct.append({
        "Campo Textual": field,
        "% > 512": pct
    })

# Visualize
plt.bar(range(len(over_512_pct)), 
        [x["% > 512"] for x in over_512_pct])
plt.title("Proportion of Texts Exceeding 512 Tokens")
plt.ylabel("%")
plt.show()
```

**Decision Points:**
- **< 5%** = Safe, minimal truncation
- **5-20%** = Consider truncation strategy
- **> 20%** = May need sliding window or hierarchical approach

---

### 5️⃣ Embedding Generation and Visualization (Cell 6)

**What it does:** Generates embeddings and creates dimensionality-reduction visualizations (UMAP, t-SNE).

#### Embedding Generation

```python
from sentence_transformers import SentenceTransformer

EMB_MODEL = "neuralmind/bert-base-portuguese-cased"
embedder = SentenceTransformer(EMB_MODEL)

# Sample data (max 2000 for computational efficiency)
SAMPLE_MAX = 2000
df_sample = df_clean.sample(n=min(SAMPLE_MAX, len(df_clean)), random_state=42)

# Create (text, field) pairs
items = []
for field in TEXT_FIELDS:
    for text in df_sample[field].astype(str):
        items.append((text, field))

texts = [text for text, _ in items]
labels = [field for _, field in items]

# Generate embeddings
embeddings = embedder.encode(texts, show_progress_bar=True, 
                            batch_size=32, convert_to_numpy=True)
```

**Embedding Characteristics:**

| Property | Value |
|----------|-------|
| **Model** | neuralmind/bert-base-portuguese-cased |
| **Embedding dimension** | 768 |
| **Language** | Portuguese-optimized |
| **Type** | Contextual word embeddings |

#### UMAP Visualization

```python
import umap

# Reduce to 2D for visualization
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, 
                    metric="cosine", random_state=42)
umap_2d = reducer.fit_transform(embeddings)

# Plot colored by text field
plt.figure(figsize=(7, 6))
for field in TEXT_FIELDS:
    mask = np.array([label == field for label in labels])
    plt.scatter(umap_2d[mask, 0], umap_2d[mask, 1], 
               s=10, label=field, alpha=0.7)
plt.legend()
plt.title("UMAP of Embeddings by Field (No Stopwords)")
plt.show()
```

**UMAP Parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| **n_neighbors** | 15 | Neighborhood size (balance local/global) |
| **min_dist** | 0.1 | Minimum cluster spacing (0.1 = loose) |
| **metric** | cosine | Distance metric (appropriate for embeddings) |

**Interpretation:**
- **Tight clusters per field** = Fields have distinct semantic characteristics
- **Mixed clusters** = Text types overlap semantically
- **Isolated points** = Outlier texts or unusual content

#### t-SNE Visualization

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Reduce to 50D with PCA first (t-SNE works better on lower dimensions)
pca = PCA(n_components=min(50, embeddings.shape[1]))
embeddings_50d = pca.fit_transform(embeddings)

# Apply t-SNE
tsne_2d = TSNE(n_components=2, perplexity=30, 
                learning_rate=200, n_iter=1000, 
                random_state=42).fit_transform(embeddings_50d)

# Plot
plt.figure(figsize=(7, 6))
for field in TEXT_FIELDS:
    mask = np.array([label == field for label in labels])
    plt.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], 
               s=10, label=field, alpha=0.7)
plt.legend()
plt.title("t-SNE of Embeddings by Field (No Stopwords)")
plt.show()
```

**t-SNE vs UMAP Comparison:**

| Aspect | UMAP | t-SNE |
|--------|------|-------|
| **Speed** | Faster | Slower |
| **Global structure** | Better preserved | Local focus |
| **Cluster density** | Reflects actual | Can distort |
| **Interpretability** | Distance meaningful | Distances misleading |
| **Reproducibility** | High | Low (stochastic) |

**Interpretation:**
- **Separated clusters** = Fields are semantically distinct
- **Overlapping regions** = Some cross-field similarity
- **Dense vs sparse** = Vocabulary consistency per field

---

## 📊 Metrics Summary and Interpretation

### Textual Metrics Output Example

```
         Campo Textual  Avg. Words  Std. Words  Avg. Chars  LexDiv (TTR)  Herdan C
0     Fato Paradigma      145.3      89.2       892.4       0.582        0.471
1     Fato Similar        156.8      92.1       967.2       0.575        0.465
2  Direito Paradigma       87.4      54.3       523.1       0.612        0.495
3  Direito Similar         92.1      58.7       551.4       0.605        0.488
```

**Interpretation Guide:**

- **Higher Avg. Words** = Longer, more complex texts
- **Higher Std. Words** = More heterogeneous text lengths
- **Higher TTR** = More varied vocabulary
- **Higher Herdan C** = Better lexical diversity (stable measure)

### Token Length Insights

```
Campo Textual          % > 512 tokens
Fato Paradigma         3.2%
Fato Similar          4.1%
Direito Paradigma     1.8%
Direito Similar       2.5%
```

**Implications:**
- Low percentage (<5%) means most texts fit BERT's 512-token limit
- Safe to use full-text embeddings without truncation
- Minimal information loss from model limitations

---

## 🎯 How to Customize

### Using Different Text Fields

```python
# Define custom fields
TEXT_FIELDS = ["your_field_1", "your_field_2", "your_field_3"]

# The rest of the code automatically adapts
```

### Changing the Embedding Model

```python
# Option 1: Another multilingual model
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Option 2: Domain-specific Portuguese model
EMB_MODEL = "DIACDE/stjiris_tjgo_diacde_sts"

# Option 3: Lightweight model (faster)
EMB_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMB_MODEL)
```

### Adjusting Sample Size

```python
# For faster analysis (less data)
SAMPLE_MAX = 500

# For comprehensive analysis (more data, slower)
SAMPLE_MAX = 5000
```

### Changing Visualization Parameters

```python
# UMAP: More local structure
reducer = umap.UMAP(n_neighbors=5, min_dist=0.01, metric="cosine")

# UMAP: More global structure
reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, metric="cosine")

# t-SNE: Faster convergence
tsne_2d = TSNE(n_components=2, perplexity=30, 
                learning_rate=300, n_iter=500, random_state=42)
```

---

## 📈 Key Findings and Insights

### What the Analysis Reveals

1. **Text Characteristics**
   - Length distribution of legal documents
   - Complexity and vocabulary diversity
   - Consistency across document types

2. **Tokenization Constraints**
   - Percentage of texts exceeding BERT's 512-token limit
   - Impact on truncation-free processing

3. **Semantic Structure**
   - Whether text fields (facts vs theses) cluster together
   - Distinctness of different document types
   - Outlier detection via visualization

4. **Preprocessing Impact**
   - Effect of stopword removal on text metrics
   - Most common removed words by field

---

## 🐛 Troubleshooting

### NLTK Stopwords Not Found

```python
# Solution: Download explicitly
import nltk
nltk.download('stopwords')
```

### Memory Error During Embedding Generation

```python
# Solution 1: Reduce sample size
SAMPLE_MAX = 1000  # Smaller sample

# Solution 2: Use lighter model
EMB_MODEL = "all-MiniLM-L6-v2"  # Smaller model

# Solution 3: Process in batches
batch_size = 16  # Smaller batches
```

### t-SNE Visualization Very Slow

```python
# Solution 1: Reduce sample size
SAMPLE_MAX = 500

# Solution 2: Reduce perplexity (faster but less accurate)
tsne_2d = TSNE(n_components=2, perplexity=10, n_iter=500)

# Solution 3: Use UMAP instead (faster)
# Remove t-SNE code and use only UMAP
```

### BERT Tokenizer Error

```python
# Solution 1: Ensure model is downloaded
# First run will download automatically, subsequent runs use cache

# Solution 2: Specify cache directory
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    cache_dir="/path/to/cache"
)
```

### File Path Issues

```python
# Use pathlib for cross-platform compatibility
from pathlib import Path

csv_path = Path("/content") / "dados_berna.csv"
df = pd.read_csv(str(csv_path))
```

---

## 📚 References

### Libraries and Tools
- [NLTK Stopwords](https://www.nltk.org/api/nltk.corpus.html#nltk.corpus.stopwords)
- [Transformers Library](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [UMAP](https://umap-learn.readthedocs.io/)
- [t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)

### Linguistic Metrics
- Type-Token Ratio (TTR)
- Herdan's C Index
- Lexical Diversity Measures

### Tokenization
- WordPiece Tokenization
- BERT Token Limits
- Subword Units

---

## 📝 Important Notes

✅ **What this notebook does well:**
- Comprehensive text corpus analysis
- Multiple dimensionality reduction visualizations
- Automatic metric computation
- Portuguese language support
- Identifies texts exceeding model constraints

⚠️ **Limitations:**
- Sampling limits full dataset analysis (memory constraints)
- t-SNE slower than UMAP (but different results)
- Stopword removal may lose some domain-specific terms
- BERT tokenization is model-specific

💡 **Best Practices:**
- Review raw text before and after stopword removal
- Check visualizations for expected clustering
- Validate metric ranges against corpus expectations
- Document any corpus-specific preprocessing decisions
- Save cleaned JSON for reproducibility

---

## 📋 File Organization

### Input Files
- `dados_berna_comparacao_fato_direito_04082025.csv`
- `dados_berna_comparacao_fato_direito_24072025.csv`

### Output Files
1. `dados_berna_comparacao_fato_direito_unificado.json` - Merged data
2. `dados_berna_comparacao_fato_direito_unificado_sem_stopwords.json` - Cleaned data
3. Various PNG charts (metrics, visualizations)

### Processed Data Flow
```
CSV Files
    ↓
Merged JSON (with stopwords)
    ↓
Cleaned JSON (without stopwords)
    ↓
Statistical Analysis + Visualizations
```

---

**Last updated:** December 2025
