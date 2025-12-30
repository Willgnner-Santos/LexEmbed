# Supervised Fine-Tuning of Sentence Transformers

## 📋 Overview

This notebook (`A2.ipynb`) performs **supervised fine-tuning of pre-trained Sentence Transformer models** on legal case similarity data. The pipeline prepares training data, trains the model using cosine similarity loss, and evaluates performance using multiple similarity metrics (Cosine, Dot Product, Euclidean, Manhattan).

### Main Flow
```
Raw Data (CSV)
    ↓
Text Cleaning & Normalization
    ↓
Data Unification and Preparation
    ↓
Dataset Creation & Train/Test/Val Split
    ↓
Model Loading and Configuration
    ↓
Fine-Tuning with Multiple Evaluators
    ↓
Model Evaluation & Saving
```

---

## 🚀 How to Use

### Prerequisites
Install required dependencies:

```bash
pip install pandas datasets sentence-transformers torch scikit-learn
```

### Basic Execution

1. **Ensure your CSV file** is in the specified path or update the file path in the code
2. **Execute the cells sequentially** in the notebook
3. **Monitor training progress** and evaluation metrics
4. **Save the fine-tuned model** for later use

---

## 📝 Detailed Section Breakdown

### 1️⃣ Text Cleaning Functions (Cell 1)

**What it does:** Removes noise and standardizes legal case text.

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

### 2️⃣ Data Loading and Cleaning (Cell 2)

**What it does:** Loads CSV file and applies cleaning functions to all text columns.

```python
df = pd.read_csv(r'/workspace/src/data/dados_berna_comparacao_fato_direito_24072025.csv', 
                  encoding='utf-8')

cols = ['fato_paradigma', 'fato_similar', 'direito_paradigma', 'direito_similar']

# Remove line breaks
df[cols] = df[cols].replace({r'[\r\n]+': ' '}, regex=True)

# Remove punctuation at beginning
df[cols] = df[cols].map(remove_pontuacao_inicio)

# Remove extra spaces
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Remove repeated punctuation
df[cols] = df[cols].map(remove_pontuacao_repetida)
```

**Expected Columns:**
| Column | Description |
|--------|-----------|
| `fato_paradigma` | Fact of the reference case |
| `fato_similar` | Fact of the similar case |
| `direito_paradigma` | Legal thesis of the reference case |
| `direito_similar` | Legal thesis of the similar case |
| `processo_paradigma` | Reference process ID |
| `processo_similar` | Similar process ID |
| `similaridade_fato` | Fact similarity score |
| `similaridade_tese` | Thesis similarity score |

---

### 3️⃣ Duplicate Removal (Cell 3)

**What it does:** Removes duplicates and groups data by process.

```python
df_unique = df.drop_duplicates(subset=['fato_similar', 'direito_similar'])

print(f"Total unique rows: {len(df_unique)}")

df = (
    df_unique.groupby('processo_paradigma')
             .apply(lambda x: x)
             .reset_index(drop=True)
)
```

---

### 4️⃣ Data Unification and Normalization (Cell 4)

**What it does:** Combines facts and theses, and normalizes similarity scores.

```python
# Merge fact + thesis for each case
df["paradigma_unificado"] = df["fato_paradigma"].astype(str) + " " + df["direito_paradigma"].astype(str)
df["similar_unificado"] = df["fato_similar"].astype(str) + " " + df["direito_similar"].astype(str)

# Calculate average similarity and normalize
df["similaridade_normalizada"] = (df["similaridade_fato"] + df["similaridade_tese"]) / 2

# Create final DataFrame for training
df = pd.DataFrame({
    "processo_paradigma": df["processo_paradigma"],
    "processo_similar": df["processo_similar"],
    "sentence1": df["paradigma_unificado"],
    "sentence2": df["similar_unificado"],
    "score": df["similaridade_normalizada"]
})
```

**Result:** Sentence pairs with similarity scores ready for model training.

---

### 5️⃣ Dataset Creation and Splitting (Cell 5) ⭐

**What it does:** Creates supervised pairs and splits data into train/test/validation sets.

```python
from datasets import Dataset, DatasetDict

# Create list with text pairs
data = []
for _, row in df.iterrows():
    data.append({
        "sentence1": row["sentence1"],
        "sentence2": row["sentence2"],
        "score": float(row["score"])
    })

print(f"{len(data)} supervised pairs created")

# Create Dataset from list
dataset = Dataset.from_list(data)

# Split into train/test/validation (80% train, 20% test, then 80% eval / 20% test)
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.2, seed=42)

train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],      # 80% of data
    'eval': test_valid['train'],             # 16% of data (80% of remaining 20%)
    'test': test_valid['test']               # 4% of data (20% of remaining 20%)
})
```

**Split Breakdown:**
- **Train:** 80% - Used to train the model
- **Eval:** 16% - Used during training to evaluate progress
- **Test:** 4% - Used for final evaluation after training

---

### 6️⃣ Model Loading and Configuration (Cell 6) ⭐⭐

**What it does:** Loads a pre-trained model and configures training parameters.

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import torch

# GPU Configuration
print(f"Device Cuda: {torch.cuda.current_device()}")  
print(f"Cuda is Available: {torch.cuda.is_available()}")

# Model configuration
model_name = "Alibaba-NLP/gte-multilingual-base"
train_batch_size = 4
num_epochs = 4
output_dir = f"output/{model_name.split('/')[-1]}-FATO_E_TESES2"

# Load the pre-trained model
model = SentenceTransformer(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
    cache_folder=os.environ["HF_HOME"],
    trust_remote_code=True
)

print(f"Model loaded: {model_name}")
print(f"Max sequence length: {model.max_seq_length}")
```

#### Available Base Models

| Model | Language Support | Speed | Quality | Use Case |
|-------|------------------|-------|---------|----------|
| **Alibaba-NLP/gte-multilingual-base** | 100+ languages | ⚡ Fast | ⭐⭐⭐⭐⭐ | Multilingual legal texts |
| **all-mpnet-base-v2** | Multilingual | Medium | ⭐⭐⭐⭐⭐ | High-quality embeddings |
| **all-MiniLM-L6-v2** | Multilingual | ⚡⚡ Very Fast | ⭐⭐⭐⭐ | Resource-constrained |
| **neuralmind/bert-base-portuguese-cased** | Portuguese | Medium | ⭐⭐⭐⭐ | Portuguese legal texts |

---

### 7️⃣ Training Configuration (Cell 6 - Part 2)

**What it does:** Sets up loss function and evaluation metrics.

#### Loss Function
```python
train_loss = losses.CosineSimilarityLoss(model=model)
```

**Why Cosine Similarity Loss?**
- Measures angle between embedding vectors
- Better for semantic similarity than Euclidean distance
- Normalized: values between -1 and 1
- Suitable for text similarity tasks

#### Evaluation Metrics

Four similarity functions are evaluated during training:

```python
# 1. Cosine Similarity (angle-based)
dev_evaluator_cosine = EmbeddingSimilarityEvaluator(
    positives=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev-cosine"
)

# 2. Dot Product (magnitude and angle)
dev_evaluator_dot = EmbeddingSimilarityEvaluator(
    main_similarity=SimilarityFunction.DOT_PRODUCT,
    name="sts-dev-dot"
)

# 3. Euclidean Distance (L2 norm)
dev_evaluator_euclidian = EmbeddingSimilarityEvaluator(
    main_similarity=SimilarityFunction.EUCLIDEAN,
    name="sts-dev-euclidian"
)

# 4. Manhattan Distance (L1 norm)
dev_evaluator_manhattan = EmbeddingSimilarityEvaluator(
    main_similarity=SimilarityFunction.MANHATTAN,
    name="sts-dev-manhattan"
)

# Combine all evaluators
dev_evaluator = SequentialEvaluator([
    dev_evaluator_dot,
    dev_evaluator_euclidian,
    dev_evaluator_manhattan,
    dev_evaluator_cosine
])
```

**Similarity Metrics Explained:**

| Metric | Formula | Range | Best For | Speed |
|--------|---------|-------|----------|-------|
| **Cosine** | $\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$ | [-1, 1] | Text similarity | ⚡⚡⚡ Fast |
| **Dot Product** | $\vec{A} \cdot \vec{B}$ | [-∞, ∞] | Dense embeddings | ⚡⚡⚡ Fast |
| **Euclidean** | $\sqrt{\sum (A_i - B_i)^2}$ | [0, ∞) | Geometric distance | ⚡⚡ Medium |
| **Manhattan** | $\sum \|A_i - B_i\|$ | [0, ∞) | High dimensions | ⚡⚡ Medium |

---

### 8️⃣ Training Arguments (Cell 6 - Part 3)

**What it does:** Defines hyperparameters for the training loop.

```python
args = SentenceTransformerTrainingArguments(
    # Output and checkpoint management
    output_dir=output_dir,
    resume_from_checkpoint=True,
    save_total_limit=2,  # Keep only 2 best checkpoints
    
    # Training parameters
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,  # Warmup 10% of training
    learning_rate=1e-5,
    
    # Precision settings
    fp16=True,   # Mixed precision (faster on NVIDIA GPUs)
    bf16=False,  # BF16 (for compatible GPUs)
    
    # Evaluation and logging
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=100
)
```

**Key Parameters Explained:**

| Parameter | Value | Explanation |
|-----------|-------|-----------|
| **num_train_epochs** | 4 | Training passes through entire dataset |
| **per_device_train_batch_size** | 4 | Samples per batch (GPU memory dependent) |
| **learning_rate** | 1e-5 | Step size for gradient descent |
| **warmup_ratio** | 0.1 | Gradually increase LR in first 10% of training |
| **eval_steps** | 100 | Evaluate every 100 training steps |
| **save_steps** | 100 | Save checkpoint every 100 steps |

---

### 9️⃣ Model Training (Cell 6 - Part 4)

**What it does:** Creates trainer and executes the fine-tuning process.

```python
from sentence_transformers.trainer import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator
)

# Start training
trainer.train()
```

**Training Process:**
1. Loads batches from training dataset
2. Forward pass: generates embeddings
3. Computes cosine similarity loss
4. Backward pass: calculates gradients
5. Updates model weights
6. Every 100 steps: evaluates on eval set
7. Saves best checkpoints

---

### 🔟 Final Evaluation and Saving (Cell 6 - Part 5)

**What it does:** Evaluates model on test set and saves the fine-tuned model.

```python
# Create test evaluators
test_evaluator_cosine = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev-cosine"
)

# ... (repeat for dot, euclidean, manhattan)

test_evaluator = SequentialEvaluator([
    test_evaluator_dot,
    test_evaluator_euclidian,
    test_evaluator_manhattan,
    test_evaluator_cosine
])

# Evaluate final model
print(test_evaluator(model))

# Save the fine-tuned model
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
```

**Output Files:**
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `sentence_transformers/` - Model artifacts
- `tokenizer.json` - Tokenizer configuration

---

## 🔄 How to Change Models

### Option 1: Use Different Base Model

Replace the model name in Cell 6:

```python
# Option A: For legal/domain-specific Portuguese
model_name = "neuralmind/bert-base-portuguese-cased"

# Option B: For multilingual with highest quality
model_name = "all-mpnet-base-v2"

# Option C: For fast lightweight embeddings
model_name = "all-MiniLM-L6-v2"

# Option D: For best multilingual
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### Option 2: Load Pre-trained Fine-tuned Model

Instead of fine-tuning from scratch:

```python
# Load previously trained model
model = SentenceTransformer(r'./output/previous_model/final')

# Continue training (optional)
trainer.train()
```

### Option 3: Use Custom Model

```python
# Load from local path
model = SentenceTransformer(r'./my_custom_model')
```

---

## 📊 Interpreting Results

### Training Metrics Output

Example output from evaluation:

```
Cosine Similarity Evaluation:
- Pearson: 0.8523
- Spearman: 0.8401

Dot Product Evaluation:
- Pearson: 0.8234
- Spearman: 0.8145
```

**What these mean:**
- **Pearson Correlation:** Measures linear relationship (0 to 1, higher is better)
- **Spearman Correlation:** Measures rank relationship (0 to 1, higher is better)
- **>0.85** = Excellent model
- **0.75-0.85** = Good model
- **<0.75** = Model needs adjustment

---

## 🎯 Common Customizations

### Increase Training Duration

```python
# Cell 6: Change number of epochs
num_epochs = 8  # Default: 4
```

### Adjust Batch Size

```python
# Cell 6: Larger batch size (requires more GPU memory)
train_batch_size = 16  # Default: 4

# Smaller batch size (slower training, less memory)
train_batch_size = 2
```

### Change Learning Rate

```python
# Cell 6: Lower learning rate (more stable, slower)
learning_rate = 2e-5  # Default: 1e-5

# Higher learning rate (faster, less stable)
learning_rate = 5e-5
```

### Adjust Warmup

```python
# Cell 6: Longer warmup period
warmup_ratio = 0.2  # Default: 0.1 (20% instead of 10%)
```

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```
Solution 1: Reduce batch size
train_batch_size = 2  # Default: 4

Solution 2: Use gradient accumulation
gradient_accumulation_steps = 2

Solution 3: Use smaller model
model_name = "all-MiniLM-L6-v2"
```

### Error: "Model not found"
```
Solution: The model will download automatically from HuggingFace
If offline, download manually:
model = SentenceTransformer('model-name')  # Downloads and caches
```

### Low Evaluation Scores
```
Possible causes and solutions:
1. Not enough training data
   → Collect more similar/dissimilar pairs

2. Training too short
   → Increase num_epochs

3. Learning rate too high
   → Decrease learning_rate

4. Poor quality labels
   → Review and clean similarity scores

5. Model unsuitable for domain
   → Try domain-specific base model
```

### Training is Very Slow
```
Solutions:
1. Use smaller model
   model_name = "all-MiniLM-L6-v2"

2. Increase batch size (if GPU memory allows)
   train_batch_size = 8

3. Reduce evaluation frequency
   eval_steps = 500  # Default: 100

4. Use data parallelism if multiple GPUs
```

---

## 📚 Using the Fine-Tuned Model

### Loading Trained Model

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('./output/gte-multilingual-base-FATO_E_TESES2/final')

# Generate embeddings
sentences = ["Your legal text here"]
embeddings = model.encode(sentences)

# Compare similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity}")
```

### Computing Similarity Between Pairs

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('./model_path')

text1 = "Description of legal case 1"
text2 = "Description of legal case 2"

# Encode both texts
emb1 = model.encode(text1, convert_to_tensor=True)
emb2 = model.encode(text2, convert_to_tensor=True)

# Compute cosine similarity
cos_sim_score = util.cos_sim(emb1, emb2)
print(f"Cosine Similarity: {cos_sim_score.item():.4f}")
```

### Batch Similarity Computation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('./model_path')

queries = ["Legal query 1", "Legal query 2"]
corpus = ["Document 1", "Document 2", "Document 3"]

# Encode all texts
query_embs = model.encode(queries, show_progress_bar=True)
corpus_embs = model.encode(corpus, show_progress_bar=True)

# Compute similarity matrix
cos_sim = util.cos_sim(query_embs, corpus_embs)
print(cos_sim)
```

---

## 📝 Important Notes

✅ **What this notebook does well:**
- Fine-tunes models on domain-specific legal data
- Evaluates multiple similarity metrics simultaneously
- Handles GPU optimization and mixed precision
- Saves best checkpoints automatically
- Provides comprehensive training logs

⚠️ **Limitations:**
- Requires GPU for efficient training (>10x slower on CPU)
- Results depend on quality of similarity labels
- May require hyperparameter tuning for new domains
- Large models require significant GPU memory

💡 **Best Practices:**
- Use at least 1,000 labeled pairs for good fine-tuning
- Ensure similarity scores are normalized [0, 1]
- Monitor evaluation metrics to detect overfitting
- Save the best checkpoint, not the last one
- Test on separate test set before deployment

---

## 📚 References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [STS Benchmark Paper](https://arxiv.org/abs/1703.05133)
- [PEFT (Parameter Efficient Fine-Tuning)](https://huggingface.co/docs/peft/)
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)

---

**Last updated:** December 2025
