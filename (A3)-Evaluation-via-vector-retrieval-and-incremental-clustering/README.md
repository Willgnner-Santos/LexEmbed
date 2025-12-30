# Evaluation via Vector Retrieval and Incremental Clustering

## 📋 Overview

This script (`A3.py`) implements **incremental clustering of legal petitions using vector embeddings and approximate nearest neighbor (ANN) search**. It uses Milvus vector database for efficient similarity-based clustering, processes legal cases incrementally, and visualizes cluster relationships as a network graph.

### Main Flow
```
Load JSON Petitions
    ↓
Initialize Milvus Vector Database
    ↓
Load Fine-Tuned Model
    ↓
For Each Petition:
  ├─ Generate Normalized Embedding
  ├─ Search Nearest Cluster Representative
  ├─ If Similar (>threshold) → Assign to Cluster
  └─ Else → Create New Cluster
    ↓
Export Results (CSV + Network Graph)
```

---

## 🚀 How to Use

### Prerequisites

Install required dependencies:

```bash
pip install pymilvus sentence-transformers numpy pandas matplotlib networkx scikit-learn
```

### Start Milvus Server

```bash
# Using Docker (recommended)
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Or start local Milvus instance
milvus standalone start
```

### Run the Script

```bash
python A3.py
```

### Expected Output

1. **Console Output:** Clustering progress and similarity scores
2. **CSV File:** `resultado_final_clusters.csv` - Complete cluster data
3. **Visualization:** `clusters_visualizacao_aprimorada.png` - Network graph

---

## 📝 Detailed Section Breakdown

### 1️⃣ Configuration Section (Lines 1-30)

**What it does:** Defines connection parameters, model paths, and output directories.

```python
# Milvus connection parameters
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Collection (table) name for clustering
COLLECTION_NAME = "peticoes_ip_final"

# Similarity threshold for cluster assignment (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.90

# Fine-tuned model path (Portuguese legal embeddings)
MODEL_PATH = r"DIACDE/stjiris_tjgo_diacde_sts"

# Output directory for results and visualizations
path_images = f"D:/sentence_transformers_FT/imgs1/{MODEL_PATH.split('/')[-1]}/"
```

**Key Parameters:**

| Parameter | Value | Explanation |
|-----------|-------|-----------|
| **MILVUS_HOST** | localhost | Milvus server hostname |
| **MILVUS_PORT** | 19530 | Default Milvus port |
| **COLLECTION_NAME** | peticoes_ip_final | Unique collection identifier |
| **SIMILARITY_THRESHOLD** | 0.90 | Min similarity to join cluster (0-1) |
| **MODEL_PATH** | Domain-specific model | Pre-trained embedding model |

---

### 2️⃣ Input Data Functions (Lines 32-80)

#### JSON to Clusters Conversion

**What it does:** Transforms JSON structure into cluster mapping.

```python
def json_to_clusters(data):
    """
    Converts:
      {proc_id: [{id:..., historia:...}, ...], ...}
    Into:
      {'1': [id, id, ...], '2': [...], ...}
    """
    clusters = {}
    for i, (proc_id, entries) in enumerate(data.items(), start=1):
        ids = [entry.get("id", proc_id) for entry in entries]
        clusters[str(i)] = ids
    return clusters
```

#### Loading Petitions

**What it does:** Loads JSON file and flattens petition structure.

```python
with open("saida3.json", "r", encoding="utf-8") as f:
    dados = json.load(f)

# Flatten into list for sequential processing
PETICOES_INICIAIS = []
for cluster in dados.values():
    for item in cluster:
        PETICOES_INICIAIS.append({
            "id": item["id"],
            "processo_numero": item["processo_numero"],
            "historia": item["historia"],
            "tese": item["tese"]
        })
```

**Input JSON Structure Expected:**
```json
{
  "processo_001": [
    {
      "id": "peticao_001",
      "processo_numero": "processo_001->processo_002",
      "historia": "Description of facts...",
      "tese": "Legal thesis..."
    }
  ]
}
```

---

### 3️⃣ Text Normalization Function (Lines 82-96)

**What it does:** Cleans and normalizes text before embedding.

```python
def text_normalizing(text):
    if isinstance(text, str):
        # Collapse repeated punctuation (>= 5) to 4 occurrences
        text = re.sub(r"([^\w\s])\1{4,}", r"\1\1\1\1", text)
        # Remove leading punctuation
        return re.sub(r"^[^\w\s]+", "", text)
    return text
```

**Example Transformations:**
- `"!!!Legal case"` → `"Legal case"`
- `"Important!!!!"` → `"Important!!!!"` (keeps 4 repetitions)
- `"  Normalized  "` → Already handled by model tokenizer

---

### 4️⃣ Milvus Connection Functions (Lines 98-135)

#### Connection and Cleanup

**What it does:** Connects to Milvus and initializes database.

```python
def conectar_e_limpar_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    print("Connection and cleanup completed.")
```

#### Collection Creation

**What it does:** Creates Milvus collection with optimized indexing.

```python
def criar_collection_milvus(dim):
    # Define schema with cluster_id and embedding vector
    fields = [
        FieldSchema(name="cluster_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    
    schema = CollectionSchema(fields, "Collection for clustering using IP metric.")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create HNSW index for fast ANN search
    index_params = {
        "metric_type": "IP",        # Inner Product (cosine for normalized vectors)
        "index_type": "HNSW",       # Hierarchical Navigable Small World
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection
```

**Index Parameters Explained:**

| Parameter | Value | Explanation |
|-----------|-------|-----------|
| **metric_type** | IP | Inner Product (= Cosine for normalized vectors) |
| **index_type** | HNSW | Hierarchical Navigable Small World (fast ANN) |
| **M** | 16 | Maximum connections per node |
| **efConstruction** | 256 | Construction parameter (higher = more accurate) |

---

### 5️⃣ Embedding Generation (Lines 137-147)

**What it does:** Generates normalized embeddings from text pairs.

```python
def gerar_vetor_normalizado(historia, tese, modelo):
    # Encode both historia and tese
    historia_vec = modelo.encode(historia)
    tese_vec = modelo.encode(tese)
    
    # Concatenate vectors (doubles dimensionality)
    combined_vec = np.concatenate([historia_vec, tese_vec])

    # L2 normalize so IP metric behaves like cosine similarity
    norm = np.linalg.norm(combined_vec)
    return (combined_vec / norm).tolist() if norm != 0 else combined_vec.tolist()
```

**Why Concatenate and Normalize?**

- **Concatenation:** Combines two semantic perspectives (facts + thesis)
- **L2 Normalization:** Converts Inner Product (IP) metric to cosine similarity
- **Formula:** $\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \vec{A}_{norm} \cdot \vec{B}_{norm}$ (for normalized vectors)

---

### 6️⃣ Main Clustering Routine (Lines 149-211) ⭐⭐

**What it does:** Incremental clustering loop - assigns petitions to clusters based on similarity.

```python
def executar_clusterizacao():
    conectar_e_limpar_milvus()
    
    # Load model
    model = SentenceTransformer(MODEL_PATH)
    final_dim = model.get_sentence_embedding_dimension() * 2
    
    # Create collection
    collection = criar_collection_milvus(final_dim)
    collection.load()

    clusters = {}
    next_cluster_id = 1

    # Process each petition
    for peticao in PETICOES_INICIAIS:
        peticao_id = peticao["id"]
        processo_numero = peticao["processo_numero"]

        # Normalize and embed
        historia = text_normalizing(peticao["historia"])
        tese = text_normalizing(peticao["tese"])
        vetor_atual = gerar_vetor_normalizado(historia, tese, model)

        found_cluster = False

        # Search in existing clusters if any
        if collection.num_entities > 0:
            search_params = {"metric_type": "IP", "params": {"ef": 64}}
            results = collection.search([vetor_atual], "embedding", search_params, limit=1)
            
            hit = results[0][0]
            score = hit.distance

            # If similar enough, assign to existing cluster
            if score >= SIMILARITY_THRESHOLD:
                clusters[hit.id].append((processo_numero, score, historia, tese))
                found_cluster = True

        # If no suitable cluster found, create new one
        if not found_cluster:
            clusters[next_cluster_id] = [(processo_numero, 1.0, historia, tese)]
            collection.insert(data=[[next_cluster_id], [vetor_atual]])
            collection.flush()
            next_cluster_id += 1

    return clusters
```

**Algorithm Logic:**

```
For each petition:
  1. Generate normalized embedding
  2. If collection has data:
     a. Search for nearest cluster representative
     b. If similarity >= threshold:
        → Add petition to cluster
     c. Else:
        → Create new cluster with this petition
  3. Else:
     → Create first cluster
```

**Why This Approach?**

- **Incremental:** Processes petitions one by one
- **Efficient:** Only stores cluster representatives (not all vectors)
- **Scalable:** HNSW index handles millions of documents
- **Deterministic:** Same order of processing = same clustering

---

### 7️⃣ Results Export (Lines 213-241)

**What it does:** Builds export dataset and saves to CSV.

```python
dataset = [
    {
        "cluster": cluster_id,
        "processo_paradigma_BERNA": processo[0].split("->")[0],
        "processo_similar_BERNA": processo[0].split("->")[1],
        "score": processo[1],
        "historia_paradigma": lista_peticoes[0][2],
        "tese_paradigma": lista_peticoes[0][3],
        "historia_similar": processo[2],
        "tese_similar": processo[3],
    }
    for cluster_id, lista_peticoes in sorted(resultado_final_clusters.items())
    for processo in lista_peticoes
]

df = pd.DataFrame(dataset)
df.to_csv(f"{path_images}resultado_final_clusters.csv", index=False)
```

**CSV Output Columns:**

| Column | Description |
|--------|-----------|
| `cluster` | Cluster ID (numeric) |
| `processo_paradigma_BERNA` | Reference process ID |
| `processo_similar_BERNA` | Similar process ID |
| `score` | Similarity score (0-1) |
| `historia_paradigma` | Facts of reference case |
| `tese_paradigma` | Thesis of reference case |
| `historia_similar` | Facts of similar case |
| `tese_similar` | Thesis of similar case |

---

### 8️⃣ Network Visualization (Lines 243-299)

**What it does:** Creates graph visualization showing cluster structure.

```python
# Build modified cluster structure (process IDs only)
resultado_final_clusters_modificado = {
    cluster_id: [processo[0].split("->")[0] for processo in lista_peticoes]
    for cluster_id, lista_peticoes in resultado_final_clusters.items()
}

# Create graph
G = nx.Graph()

for cluster_id, processos in resultado_final_clusters_modificado.items():
    unique_processos = list(set(processos))
    
    # Add nodes with cluster attribute
    for processo in unique_processos:
        G.add_node(processo, cluster=cluster_id)
    
    # Connect all nodes in cluster (clique)
    for i in range(len(unique_processos)):
        for j in range(i + 1, len(unique_processos)):
            G.add_edge(unique_processos[i], unique_processos[j])

# Assign colors per cluster
num_clusters = len(resultado_final_clusters_modificado)
colors = cm.get_cmap("tab20", max(num_clusters, 20))
cluster_colors = {cid: colors(cid % 20) for cid in resultado_final_clusters_modificado.keys()}
node_colors = [cluster_colors[G.nodes[n]["cluster"]] for n in G.nodes]

# Layout and render
pos = nx.spring_layout(G, seed=42, k=1.0, iterations=100)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=6, font_color="black")

plt.savefig(f"{path_images}clusters_visualizacao_aprimorada.png", 
            format="png", dpi=300, bbox_inches="tight")
```

**Visualization Features:**
- **Nodes:** Legal cases (processes)
- **Edges:** Cases in same cluster
- **Colors:** Different color per cluster
- **Layout:** Spring layout for visual separation
- **Resolution:** 300 DPI for reporting quality

---

## 🔧 Configuration Guide

### Adjusting Similarity Threshold

```python
# More conservative (larger clusters)
SIMILARITY_THRESHOLD = 0.85

# More strict (smaller clusters)
SIMILARITY_THRESHOLD = 0.95
```

**Impact:**
- **Lower threshold** → Larger clusters, fewer clusters total
- **Higher threshold** → Smaller clusters, more clusters total
- **Typical range:** 0.80-0.95

### Changing the Model

```python
# Option 1: Use different fine-tuned model
MODEL_PATH = r"path/to/your/model"

# Option 2: Use HuggingFace model
MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Option 3: Use domain-specific model
MODEL_PATH = "neuralmind/bert-base-portuguese-cased"
```

### Adjusting HNSW Parameters

```python
# More memory, faster search
index_params = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 32, "efConstruction": 512},  # Larger values
}

# Less memory, slower search
index_params = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 128},   # Smaller values
}
```

**HNSW Parameters:**
- **M:** Connections per node (default: 16, range: 5-48)
- **efConstruction:** Construction effort (default: 256, range: 64-512)
- Higher values = more accurate but slower

---

## 🎯 How Incremental Clustering Works

### Step-by-Step Example

**Initial State:** No clusters exist

```
Petition 1: "Legal facts A..."
  ├─ Generate embedding
  ├─ Milvus has 0 entities
  └─ CREATE Cluster 1 with Petition 1
```

**After Petition 1:**
```
Clusters:
  Cluster 1: [Petition 1] (representative: embedding_1)
```

**Petition 2: "Similar facts A'..."**

```
Petition 2: "Similar facts A'..."
  ├─ Generate embedding
  ├─ Search Milvus: Find nearest = Cluster 1 (similarity: 0.92)
  ├─ 0.92 >= 0.90 (threshold)? YES
  └─ ADD to Cluster 1
```

**After Petition 2:**
```
Clusters:
  Cluster 1: [Petition 1, Petition 2]
  (representative still: embedding_1, no update)
```

**Petition 3: "Completely different legal matter..."**

```
Petition 3: "Different matter..."
  ├─ Generate embedding
  ├─ Search Milvus: Find nearest = Cluster 1 (similarity: 0.65)
  ├─ 0.65 >= 0.90 (threshold)? NO
  └─ CREATE Cluster 2 with Petition 3
```

**Final Result:**
```
Clusters:
  Cluster 1: [Petition 1, Petition 2]
  Cluster 2: [Petition 3]
```

---

## 📊 Understanding Output

### Console Output Example

```
--- Processing Petition: peticao_001 ---
[Milvus Search] Nearest candidate: Cluster 1 | Similarity score: 92.5000%
--> Assigned: Petition 'peticao_001' appended to Cluster 1.

--- Processing Petition: peticao_002 ---
[Milvus Search] Nearest candidate: Cluster 1 | Similarity score: 65.3000%
--> Action: Created new Cluster 2 using 'peticao_002' as representative.
```

### CSV Output Example

| cluster | processo_paradigma_BERNA | processo_similar_BERNA | score | historia_paradigma | tese_paradigma |
|---------|-------------------------|------------------------|-------|-------------------|----------------|
| 1 | PROC_001 | PROC_002 | 0.925 | Facts of case 1... | Thesis 1... |
| 1 | PROC_001 | PROC_003 | 0.891 | Facts of case 1... | Thesis 1... |
| 2 | PROC_004 | PROC_005 | 0.812 | Facts of case 4... | Thesis 4... |

---

## 🐛 Troubleshooting

### Error: "Connection refused" (Milvus)

```
Solution 1: Start Milvus server
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

Solution 2: Check connection parameters
MILVUS_HOST = "localhost"  # or IP address
MILVUS_PORT = "19530"
```

### Error: "Model not found"

```
Solution 1: Download model first
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("DIACDE/stjiris_tjgo_diacde_sts")

Solution 2: Use local model path
MODEL_PATH = r"./local_models/my_model"
```

### Memory Issues with Large Datasets

```
Solutions:
1. Process in batches (modify loop to process N petitions)
2. Reduce embedding dimension (use smaller model)
3. Use lighter model: "all-MiniLM-L6-v2"
4. Clear intermediate data periodically
```

### Slow Clustering Process

```
Solutions:
1. Reduce SIMILARITY_THRESHOLD (fewer cluster searches)
2. Decrease efConstruction in HNSW (faster but less accurate)
3. Use batch processing instead of incremental
4. Process on GPU if available (modify model loading)
```

### Inconsistent Results with Different Runs

This is **expected behavior** for HNSW index. For reproducible results:

```python
# Set seed
np.random.seed(42)

# Use deterministic layout
pos = nx.spring_layout(G, seed=42, k=1.0, iterations=100)
```

---

## 📈 Performance Metrics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Embedding generation** | O(n) | Linear with number of petitions |
| **ANN search** | O(log n) | HNSW approximate search |
| **Clustering** | O(n × log n) | For all n petitions |
| **Graph creation** | O(c²) | c = max cluster size |

### Storage Estimates

- **Vector database:** ~400 bytes per embedding (768-dim) × number of clusters
- **CSV export:** ~2-5 KB per cluster record
- **Graph visualization:** PNG size varies with cluster count

---

## 💡 Best Practices

✅ **Do:**
- Test with small subset first before running full dataset
- Monitor similarity scores to validate threshold choice
- Save clustering results before creating visualizations
- Use appropriate model for domain (legal-specific preferred)
- Document threshold and model choice in outputs

❌ **Don't:**
- Set threshold too low (<0.70) - creates meaningless clusters
- Set threshold too high (>0.99) - too many single-element clusters
- Modify cluster order after creation (breaks reproducibility)
- Run without backing up input JSON file
- Use random seed changes during production runs

---

## 📚 References

- [Milvus Vector Database](https://milvus.io/)
- [HNSW Algorithm](https://arxiv.org/abs/1802.02413)
- [Sentence Transformers](https://www.sbert.net/)
- [NetworkX Documentation](https://networkx.org/)

---

## 📝 Input/Output Summary

### Input Files Required
- `saida3.json` - JSON with petitions grouped by process

### Output Files Generated
1. `resultado_final_clusters.csv` - Cluster data export
2. `clusters_visualizacao_aprimorada.png` - Network visualization

### Key Outputs
- Console logs with similarity scores
- Cluster membership statistics
- Graph statistics (nodes, edges, density)

---

**Last updated:** December 2025
