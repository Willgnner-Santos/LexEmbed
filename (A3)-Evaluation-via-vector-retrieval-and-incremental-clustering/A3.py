import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
import json
import os

# --- 1. GENERAL CONFIG ---
# Defined Milvus connection parameters (local default).
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Set a clean collection name to avoid conflicts with prior experiments.
COLLECTION_NAME = "peticoes_ip_final"

# Defined the similarity threshold used to decide whether a petition joins an existing cluster.
SIMILARITY_THRESHOLD = 0.90

# Defined the SentenceTransformer model path (Portuguese legal STS fine-tuned).
MODEL_PATH = r"DIACDE/stjiris_tjgo_diacde_sts"

# Created an output folder to store generated artifacts (CSV + cluster visualization image).
path_images = f"D:/sentence_transformers_FT/imgs1/{MODEL_PATH.split('/')[-1]}/"
if not os.path.exists(path_images):
    os.makedirs(path_images)

# --- 2. INPUT DATA ---
def json_to_clusters(data):
    """
    Converted an input mapping like:
      {proc_id: [ {id:..., historia:...}, ... ], ...}
    into a simplified cluster-index mapping like:
      {'1': [id, id, ...], '2': [...], ...}
    Repetitions were preserved (one id per object in the list).
    """
    clusters = {}
    for i, (proc_id, entries) in enumerate(data.items(), start=1):
        # Extracted "id" from each entry; fell back to proc_id if "id" was missing.
        ids = [entry.get("id", proc_id) for entry in entries]
        clusters[str(i)] = ids
    return clusters

# Loaded the JSON file containing petitions grouped by process.
with open("saida3.json", "r", encoding="utf-8") as f:
    dados = json.load(f)

# Built an "old clusters" structure from the JSON (kept for reference/diagnostics).
clusters_antigos = json_to_clusters(dados)

import re

def text_normalizing(text):
    # Implemented lightweight cleanup to reduce noisy punctuation sequences.
    if isinstance(text, str):
        # Collapsed repeated punctuation (>= 5) down to 4 occurrences.
        text = re.sub(r"([^\w\s])\1{4,}", r"\1\1\1\1", text)
        # Removed punctuation-only prefix at the beginning of the string.
        return re.sub(r"^[^\w\s]+", "", text)
    return text

# Flattened the JSON structure into a list of petitions to be processed sequentially.
PETICOES_INICIAIS = []
for cluster in dados.values():
    for item in cluster:
        PETICOES_INICIAIS.append(
            {
                "id": item["id"],
                "processo_numero": item["processo_numero"],  # Added process number for later export.
                "historia": item["historia"],
                "tese": item["tese"],
            }
        )

# --- 3. HELPER FUNCTIONS ---
def conectar_e_limpar_milvus():
    # Connected to Milvus and dropped the existing collection to start fresh.
    print("--- CONNECTING AND CLEANING MILVUS ---")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    print("Connection and cleanup completed.")

def criar_collection_milvus(dim):
    """
    Created a Milvus collection configured for cosine-like similarity using:
    - Normalized vectors
    - Inner Product (IP) metric
    """
    # Defined schema: cluster_id as primary key + embedding vector field.
    fields = [
        FieldSchema(name="cluster_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "Collection for clustering using IP metric.")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Created an HNSW index optimized for approximate nearest neighbor search.
    index_params = {
        "metric_type": "IP",      # Used inner product to score similarity.
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Collection '{COLLECTION_NAME}' created with HNSW index and IP metric.")
    return collection

def gerar_vetor_normalizado(historia, tese, modelo):
    # Generated embeddings for "historia" and "tese", concatenated them, and L2-normalized the final vector.
    historia_vec = modelo.encode(historia)
    tese_vec = modelo.encode(tese)
    combined_vec = np.concatenate([historia_vec, tese_vec])

    # Performed L2 normalization so IP behaves like cosine similarity.
    norm = np.linalg.norm(combined_vec)
    return (combined_vec / norm).tolist() if norm != 0 else combined_vec.tolist()

# --- 4. MAIN CLUSTERING ROUTINE (IP-BASED LOGIC) ---
def executar_clusterizacao():
    # Initialized Milvus environment and collection.
    conectar_e_limpar_milvus()

    print("\n--- LOADING LANGUAGE MODEL ---")
    model = SentenceTransformer(MODEL_PATH)

    # Doubled the vector dimension because embeddings were concatenated (historia + tese).
    final_dim = model.get_sentence_embedding_dimension() * 2

    collection = criar_collection_milvus(final_dim)
    collection.load()

    # Initialized cluster storage and incremental cluster IDs.
    clusters = {}
    next_cluster_id = 1

    # Iterated through all petitions and assigned each one to a cluster.
    for peticao in PETICOES_INICIAIS:
        peticao_id = peticao["id"]
        processo_numero = peticao["processo_numero"]

        print(f"\n--- Processing Petition: {peticao_id} ---")

        # Normalized texts before embedding.
        historia = text_normalizing(peticao["historia"])
        tese = text_normalizing(peticao["tese"])

        # Built the normalized embedding used for ANN search.
        vetor_atual = gerar_vetor_normalizado(historia, tese, model)

        found_cluster = False

        # If at least one cluster representative exists, searched the nearest representative in Milvus.
        if collection.num_entities > 0:
            search_params = {"metric_type": "IP", "params": {"ef": 64}}
            results = collection.search([vetor_atual], "embedding", search_params, limit=1)

            # Took the best match (top-1).
            hit = results[0][0]

            # With IP metric and normalized vectors, hit.distance is treated as similarity score.
            score = hit.distance

            print(
                f"[Milvus Search] Nearest candidate: Cluster {hit.id} | Similarity score: {score:.6%}"
            )

            # If score surpassed the threshold, assigned the petition to the matched cluster.
            if score >= SIMILARITY_THRESHOLD:
                clusters[hit.id].append((processo_numero, score, historia, tese))
                print(f"--> Assigned: Petition '{peticao_id}' appended to Cluster {hit.id}.")
                found_cluster = True

        # If no suitable cluster was found, created a new cluster and inserted its representative vector.
        if not found_cluster:
            print(
                f"--> Action: Created new Cluster {next_cluster_id} using '{peticao_id}' as representative."
            )
            clusters[next_cluster_id] = [(processo_numero, 1.0, historia, tese)]

            # Inserted the representative embedding with the cluster_id as the primary key.
            collection.insert(data=[[next_cluster_id], [vetor_atual]])
            collection.flush()

            next_cluster_id += 1

    return clusters

from sklearn.metrics import silhouette_score
import numpy as np

# --- 5. RUN ---
if __name__ == "__main__":
    # Executed the clustering routine and produced the final cluster mapping.
    resultado_final_clusters = executar_clusterizacao()

    print("\n\n==============================================")
    print("--- FINAL SIMILARITY GROUPS (CLUSTERS) ---")
    print("==============================================")

    # Kept the print loop structure for optional debugging/inspection.
    for cluster_id, lista_peticoes in sorted(
        resultado_final_clusters.items(), key=lambda x: (isinstance(x[0], int), x[0])
    ):
        # Printing was intentionally skipped (useful for large outputs).
        pass

# Built an export dataset containing cluster structure and representative text fields.
dataset = [
    {
        "cluster": cluster_id,
        "processo_paradigma_BERNA": processo[0].split("->")[0],
        "processo_similar_BERNA": processo[0].split("->")[1],
        "score": processo[1],
        "historia_paradigma": lista_peticoes[0][2],  # Used the first element as the paradigm text.
        "tese_paradigma": lista_peticoes[0][3],
        "historia_similar": processo[2],
        "tese_similar": processo[3],
    }
    for cluster_id, lista_peticoes in sorted(resultado_final_clusters.items())
    for processo in lista_peticoes
]

import pandas as pd

# Exported the cluster dataset as CSV for downstream analysis.
df = pd.DataFrame(dataset)
df.to_csv(f"{path_images}resultado_final_clusters.csv", index=False)

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

# Reduced the cluster structure to only process identifiers for graph visualization.
resultado_final_clusters_modificado = {
    cluster_id: [processo[0].split("->")[0] for processo in lista_peticoes]
    for cluster_id, lista_peticoes in resultado_final_clusters.items()
}

print("\n\n==============================================")
print("--- FINAL SIMILARITY GROUPS (MODIFIED VIEW) ---")
print("==============================================")

# Printed cluster membership sizes for quick diagnostics.
for cluster_id, lista_peticoes in sorted(
    resultado_final_clusters_modificado.items(), key=lambda x: (isinstance(x[0], int), x[0])
):
    print(
        f"Cluster {cluster_id}: {lista_peticoes} | processes in cluster: {len(lista_peticoes)}"
    )

# Created an undirected graph where nodes are processes and edges connect processes in the same cluster.
G = nx.Graph()

# Added nodes and intra-cluster edges (cliques) to emphasize cluster structure.
for cluster_id, processos in resultado_final_clusters_modificado.items():
    unique_processos = list(set(processos))

    # Added each process as a node with its cluster attribute.
    for processo in unique_processos:
        G.add_node(processo, cluster=cluster_id)

    # Connected all nodes within a cluster (clique).
    for i in range(len(unique_processos)):
        for j in range(i + 1, len(unique_processos)):
            G.add_edge(unique_processos[i], unique_processos[j])

# Assigned colors per cluster using a categorical colormap.
num_clusters = len(resultado_final_clusters_modificado)
colors = cm.get_cmap("tab20", max(num_clusters, 20))
cluster_colors = {cid: colors(cid % 20) for cid in resultado_final_clusters_modificado.keys()}
node_colors = [cluster_colors[G.nodes[n]["cluster"]] for n in G.nodes]

# Generated node labels (kept full IDs for traceability).
node_labels = {node: node for node in G.nodes()}

# --- PLOTTING CONFIG ---
plt.figure(figsize=(25, 18))

# Computed a spring layout to spread clusters and reduce overlaps.
pos = nx.spring_layout(G, seed=42, k=1.0, iterations=100)

# Rendered nodes, edges, and labels.
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray")
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_color="black")

plt.title("Cluster Visualization (Enhanced)", size=20)
plt.axis("off")
plt.tight_layout()

# Saved the figure at high resolution for reporting and inspection.
plt.savefig(
    f"{path_images}clusters_visualizacao_aprimorada.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
