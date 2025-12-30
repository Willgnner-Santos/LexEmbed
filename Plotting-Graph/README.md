# Plotting Graph - Experimental Results Visualization

## 📋 Overview

This notebook (`Plotting-Graph.ipynb`) generates **publication-quality visualizations of clustering experiments** comparing different embedding models with various similarity metrics and fine-tuning configurations. It creates professional academic figures suitable for research papers and presentations.

### Main Features
```
Load Experimental Results
    ↓
Configure Publication-Quality Style
    ↓
Create Grouped Bar Charts
    ↓
Annotate Values & Format Axes
    ↓
Export High-Resolution Figures (300 DPI)
```

---

## 🚀 How to Use

### Prerequisites

Install required dependencies:

```bash
pip install numpy matplotlib
```

### Basic Execution

1. **Prepare your data** - Define cluster counts from experiments
2. **Run the notebook** - Execute cells sequentially
3. **Customize styling** - Adjust colors, fonts, and sizes as needed
4. **Export figures** - High-resolution PNG files for publications

### Output

- `fig_clusters_holdout_negrito_sem_grade.png` - Main cluster comparison figure (300 DPI)

---

## 📝 Detailed Section Breakdown

### 1️⃣ Global Style Configuration (Cell 1)

**What it does:** Sets up publication-quality styling for all plots.

```python
plt.rcParams.update({
    "font.size": 10,                    # Standard academic font size
    "font.family": "serif",             # Use serif fonts (Times-like)
    "text.color": "black",              # Black text for all elements
    "axes.labelcolor": "black",         # Black axis labels
    "xtick.color": "black",             # Black tick marks (X axis)
    "ytick.color": "black",             # Black tick marks (Y axis)
    "axes.spines.top": False,           # Remove top border
    "axes.spines.right": False,         # Remove right border
    "figure.dpi": 300,                  # High resolution (publication-ready)
})
```

**Why These Settings?**

| Setting | Purpose | Academic Standard |
|---------|---------|-------------------|
| **serif font** | Professional appearance | IEEE, ACM, Springer |
| **font.size = 10** | Readable in printed papers | Standard for 1-column figures |
| **black text** | High contrast, prints well | B&W printing compatible |
| **no top/right spines** | Clean, modern look | Tufte data visualization principle |
| **300 DPI** | Publication quality | Minimum for print journals |

---

### 2️⃣ Experimental Data (Cell 1 - Part 2)

**What it does:** Defines the embedding models and cluster counts from experiments.

```python
# List of evaluated models
models = [
    "STJ Iris",           # Fine-tuned legal model
    "BERTimbau Large",    # Portuguese BERT (large)
    "BERTimbau Base",     # Portuguese BERT (base)
    "LegalBert-pt",       # Legal BERT for Portuguese
    "RoBERTaLexPT-base",  # RoBERTa for legal Portuguese
    "JurisBERT"           # Legal-specific BERT
]

# Cluster counts: Cosine Similarity + Fine-Tuned (FT)
clusters_cos_ft = [9, 9, 9, 23, 2, 21]

# Cluster counts: Cosine Similarity + No Fine-Tuning
clusters_cos_noft = [9, 9, 9, 12, 1, 42]

# Cluster counts: Inner Product (IP) + Fine-Tuned (FT)
clusters_pi_ft = [26, 24, 22, 23, 2, 21]

# Cluster counts: Inner Product (IP) + No Fine-Tuning
clusters_pi_noft = [44, 2, 8, 12, 1, 42]
```

**Data Structure Explanation:**

```
Experimental Dimensions:
├─ Similarity Metric (2): Cosine (Cos) or Inner Product (IP)
├─ Fine-Tuning (2): With FT or Without FT
└─ Models (6): Different embedding models

Total Combinations: 2 × 2 × 6 = 24 experimental settings
```

**Data Interpretation:**

| Model | Cos+FT | Cos+NoFT | IP+FT | IP+NoFT |
|-------|--------|----------|-------|---------|
| STJ Iris | 9 | 9 | 26 | 44 |
| BERTimbau Large | 9 | 9 | 24 | 2 |
| BERTimbau Base | 9 | 9 | 22 | 8 |
| LegalBert-pt | 23 | 12 | 23 | 12 |
| RoBERTaLexPT-base | 2 | 1 | 2 | 1 |
| JurisBERT | 21 | 42 | 21 | 42 |

**Key Observations:**
- **Fine-tuning effect:** Compare FT vs NoFT columns
- **Metric effect:** Compare Cos vs IP rows
- **Model performance:** Compare across rows for given settings

---

### 3️⃣ Helper Function: Bar Annotation (Cell 1 - Part 3)

**What it does:** Automatically adds value labels above bars in plots.

```python
def annotate_bars(ax, bars, fmt="{:.0f}", rotation=0):
    """
    Add numeric labels above bar chart bars.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object containing the bars
    bars : BarContainer
        The bar objects returned from ax.bar()
    fmt : str
        Format string for numbers (e.g., "{:.0f}" for integers)
    rotation : int
        Text rotation angle in degrees
    """
    for bar in bars:
        height = bar.get_height()  # Get bar height
        ax.annotate(
            fmt.format(height),           # Format the value
            xy=(bar.get_x() + bar.get_width() / 2, height),  # Position at bar top
            xytext=(0, 3),                # Offset upward by 3 points
            textcoords="offset points",
            ha="center",                  # Center horizontally
            va="bottom",                  # Align bottom of text to position
            rotation=rotation,
            fontsize=8,
            fontweight="bold",            # Make values stand out
            color="black",
        )
```

**Visual Effect:**

```
         9           ← Annotated value
    ┌────────┐
    │        │
    │        │ Bar height = 9
    │        │
    └────────┘
   STJ Iris
```

**Usage Example:**
```python
annotate_bars(ax, bars_cos_ft)   # Adds labels to Cos+FT bars
```

---

### 4️⃣ Main Plot: Grouped Bar Chart (Cell 1 - Part 4) ⭐

**What it does:** Creates grouped bar chart comparing all conditions.

#### Figure Setup
```python
# Create figure with specific size (width=6in, height=5in)
fig, ax = plt.subplots(figsize=(6, 5))
```

**Figure Size Selection:**
- **6×5 inches** is standard for 1-column journals
- **Aspect ratio ~1.2:1** ensures readability when scaled to column width

#### Position Calculation
```python
# Create x-positions for models
x = np.arange(len(models))  # [0, 1, 2, 3, 4, 5]

# Bar width for each condition
width = 0.18  # 18% of inter-model spacing

# Positions within each model group:
#   Cos+FT:    -1.5 × width = -0.27
#   Cos+NoFT:  -0.5 × width = -0.09
#   IP+FT:     +0.5 × width = +0.09
#   IP+NoFT:   +1.5 × width = +0.27
```

**Visual Layout:**

```
Model spacing:      ←─ 1 unit ─→
                    
Bar arrangement:    | | | | |
                    |-|-|-|-|
                    | | | | |
                    
Bar width:          0.18 units each
Total width used:   0.18 × 4 = 0.72 units

Spacing between groups: 1 - 0.72 = 0.28 units
```

#### Creating Bar Groups
```python
# Each ax.bar() creates one set of bars (one condition)
bars_cos_ft   = ax.bar(x - 1.5*width, clusters_cos_ft,   width, label="Cos, FN")
bars_cos_noft = ax.bar(x - 0.5*width, clusters_cos_noft, width, label="Cos, without FN")
bars_pi_ft    = ax.bar(x + 0.5*width, clusters_pi_ft,    width, label="IP, FN")
bars_pi_noft  = ax.bar(x + 1.5*width, clusters_pi_noft,  width, label="IP, without FN")
```

**Color Assignment:** matplotlib automatically assigns distinct colors to each bar group.

#### Axis Configuration
```python
# Y-axis label
ax.set_ylabel("Number of clusters", fontweight="bold")

# X-axis positions and labels
ax.set_xticks(x)  # Set tick positions at [0, 1, 2, 3, 4, 5]
ax.set_xticklabels(models, rotation=20, ha="right")

# Remove grid for cleaner appearance
ax.grid(False)
```

**Rotation Impact:**
- **20° rotation** prevents label overlap
- **ha="right"** (right-aligned) aligns rotated labels properly
- Improves readability for long model names

#### Styling Refinements
```python
# Emphasize model names on x-axis
for label in ax.get_xticklabels():
    label.set_fontweight("medium")

# Annotate all bar groups with values
annotate_bars(ax, bars_cos_ft)
annotate_bars(ax, bars_cos_noft)
annotate_bars(ax, bars_pi_ft)
annotate_bars(ax, bars_pi_noft)

# Add legend to identify conditions
ax.legend()

# Adjust spacing to prevent label clipping
plt.tight_layout()
```

#### Export and Display
```python
# Save as high-resolution PNG
plt.savefig("fig_clusters_holdout_negrito_sem_grade.png", dpi=300, bbox_inches="tight")

# Display in notebook
plt.show()
```

**Output File Details:**
- **Format:** PNG (lossless, good for prints)
- **DPI:** 300 (publication standard)
- **Filename:** `fig_clusters_holdout_negrito_sem_grade.png`
  - "negrito" = bold
  - "sem grade" = no grid

---

## 🎨 Customization Guide

### Changing Colors

```python
# Option 1: Set colors explicitly when creating bars
bars_cos_ft = ax.bar(x - 1.5*width, clusters_cos_ft, width, 
                     label="Cos, FN", color="#1f77b4")  # Blue
bars_cos_noft = ax.bar(x - 0.5*width, clusters_cos_noft, width,
                       label="Cos, without FN", color="#ff7f0e")  # Orange

# Option 2: Use a custom colormap
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]  # Red, Blue, Green, Orange
for i, (bar_group, color) in enumerate(zip([bars_cos_ft, bars_cos_noft, bars_pi_ft, bars_pi_noft], colors)):
    for bar in bar_group:
        bar.set_color(color)
```

### Adjusting Figure Size

```python
# Larger figure for presentation
fig, ax = plt.subplots(figsize=(10, 7))

# Smaller figure for space-constrained journals
fig, ax = plt.subplots(figsize=(4, 3))
```

**Common Publication Sizes:**
- **Single column:** 3.5-4.5 inches wide
- **Double column:** 6-7 inches wide
- **Height:** Usually 0.6-0.8 × width

### Changing Font Sizes

```python
plt.rcParams.update({
    "font.size": 12,                    # Larger base font
    "axes.labelsize": 14,               # Axis labels
    "axes.titlesize": 16,               # Title
    "xtick.labelsize": 10,              # X-axis tick labels
    "ytick.labelsize": 10,              # Y-axis tick labels
    "legend.fontsize": 10,              # Legend
})
```

### Adding Title and Improving Layout

```python
# Add title
ax.set_title("Cluster Count Comparison Across Models and Methods", 
             fontweight="bold", fontsize=14, pad=20)

# Add subtitle or note
fig.text(0.5, 0.02, "FN = Fine-tuned, Cos = Cosine similarity, IP = Inner Product",
         ha='center', fontsize=9, style='italic')

# Adjust spacing
plt.subplots_adjust(top=0.95, bottom=0.15)
```

### Creating Multiple Subplots

```python
# Create 2×2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot on each subplot
axes[0, 0].bar(x, clusters_cos_ft, width, label="Cos, FN")
axes[0, 1].bar(x, clusters_cos_noft, width, label="Cos, without FN")
axes[1, 0].bar(x, clusters_pi_ft, width, label="IP, FT")
axes[1, 1].bar(x, clusters_pi_noft, width, label="IP, without FT")

# Label each subplot
for idx, ax in enumerate(axes.flat):
    ax.set_title(f"Condition {idx+1}")
    ax.set_xticklabels(models, rotation=20, ha="right")
```

---

## 📊 Data Visualization Best Practices

### Do's ✅

- **Use consistent colors** for same categories across figures
- **Include value labels** for precise reading
- **Remove unnecessary elements** (gridlines, top/right spines)
- **Rotate long labels** to prevent overlap
- **Use white space** effectively
- **Export at 300 DPI** for print publications
- **Test B&W printing** to ensure accessibility

### Don'ts ❌

- **Don't use 3D effects** - they obscure data
- **Don't use non-standard fonts** - may not embed in PDFs
- **Don't use bright colors** without purpose
- **Don't forget units** on axis labels
- **Don't place legend over data**
- **Don't use low DPI** (<150) for prints

---

## 📈 Interpreting the Visualization

### Key Metrics to Compare

**1. Effect of Fine-Tuning:**
```
For each model and metric:
  Difference = |FT clusters - NoFT clusters|
  
Higher difference → Stronger FT effect
```

**2. Effect of Similarity Metric:**
```
For each model and FT condition:
  Difference = |IP clusters - Cosine clusters|
  
Higher difference → Metric choice matters more
```

**3. Model Consistency:**
```
Models with less variation across conditions
  → More robust and reliable
  
Models with high variation
  → Sensitive to experimental parameters
```

### Example Analysis

From the data:

| Aspect | Observation |
|--------|-------------|
| **Most stable model** | STJ Iris (9, 9, 26, 44) - varies only with metric |
| **Most variable model** | JurisBERT (21, 42, 21, 42) - high FT effect |
| **Metric effect** | IP produces higher cluster counts than Cos |
| **FT effectiveness** | Mixed: helps some models, hurts others |

---

## 🔄 Adapting for Different Data

### Using Different Models

```python
# Replace the models list
models = [
    "Model A",
    "Model B", 
    "Model C",
    "Model D"
]

# Adjust data accordingly
clusters_cos_ft = [10, 15, 8, 12]
clusters_cos_noft = [20, 25, 18, 22]
clusters_pi_ft = [5, 8, 3, 6]
clusters_pi_noft = [15, 18, 10, 14]
```

### Adding More Experimental Conditions

```python
# Add more bar groups
width = 0.12  # Smaller width to fit more bars

bars_1 = ax.bar(x - 1.5*width, data_1, width, label="Condition 1")
bars_2 = ax.bar(x - 0.5*width, data_2, width, label="Condition 2")
bars_3 = ax.bar(x + 0.5*width, data_3, width, label="Condition 3")
bars_4 = ax.bar(x + 1.5*width, data_4, width, label="Condition 4")
bars_5 = ax.bar(x + 2.5*width, data_5, width, label="Condition 5")  # New

# Note: May need to adjust spacing and font sizes for readability
```

### Plotting Different Types of Data

```python
# For performance metrics (0-100 scale)
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)", fontweight="bold")

# For timing data (seconds)
ax.set_ylabel("Execution Time (s)", fontweight="bold")

# For proportional data
ax.set_ylim(0, 1)
ax.set_ylabel("Proportion", fontweight="bold")
```

---

## 🐛 Troubleshooting

### Overlapping Labels on X-Axis

```python
# Solution 1: Increase rotation
ax.set_xticklabels(models, rotation=45, ha="right")

# Solution 2: Decrease font size
ax.set_xticklabels(models, fontsize=8)

# Solution 3: Increase figure width
fig, ax = plt.subplots(figsize=(10, 5))  # Wider
```

### Bars Look Too Narrow or Too Wide

```python
# Too narrow: increase width
width = 0.25  # Was 0.18

# Too wide: decrease width
width = 0.12  # Was 0.18

# Adjust bar positions if changing width
bars_cos_ft = ax.bar(x - 1.5*width, clusters_cos_ft, width, label="Cos, FN")
```

### Legend Overlaps Data

```python
# Move legend outside plot area
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# Or place it in empty corner
ax.legend(loc="lower right")

# Or above the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
```

### Annotations Not Visible

```python
# Increase annotation font size
fontsize=10  # Was 8

# Ensure text color contrasts with background
color="white"  # Or "black"

# Increase offset from bars
xytext=(0, 5)  # Was (0, 3)
```

### Export Quality Issues

```python
# For better quality
plt.savefig("figure.png", dpi=300, bbox_inches="tight", facecolor="white")

# For PDF (vector format, preferred for papers)
plt.savefig("figure.pdf", bbox_inches="tight", facecolor="white")

# For EPS (older format, still used by some journals)
plt.savefig("figure.eps", bbox_inches="tight", facecolor="white")
```

---

## 📚 Publication Guidelines

### Journal Requirements

| Journal Type | DPI | Format | Color |
|--------------|-----|--------|-------|
| **IEEE/ACM** | 300+ | EPS/PDF | B&W + 1 color |
| **Springer** | 300+ | PDF | RGB or CMYK |
| **Elsevier** | 300+ | EPS/TIFF | RGB |
| **Conferences** | 150+ | PDF | Any |

### Tips for Journal Submission

1. **Save in multiple formats:**
   ```python
   plt.savefig("figure.png", dpi=300, bbox_inches="tight")
   plt.savefig("figure.pdf", bbox_inches="tight")
   ```

2. **Verify readability at actual size:**
   - Print or view at 100% zoom
   - Ensure text is readable (typically >8pt)

3. **Check compatibility:**
   - Fonts must be embedded (PDF) or standard (PNG)
   - Test on different platforms/viewers

4. **Remove unnecessary elements:**
   - No watermarks or logos
   - No "Figure X" caption (usually added by journal)

---

## 💡 Advanced Customizations

### Adding Statistical Annotations

```python
from scipy import stats

# Example: Add significance markers
p_value = 0.05
if p_value < 0.001:
    marker = "***"
elif p_value < 0.01:
    marker = "**"
elif p_value < 0.05:
    marker = "*"

ax.text(x_pos, y_pos, marker, ha="center", fontsize=12)
```

### Creating Heatmap-Style Bar Chart

```python
# Color bars by value intensity
norm = plt.Normalize(vmin=min_value, vmax=max_value)
colors = plt.cm.RdYlGn(norm(clusters_cos_ft))

bars = ax.bar(x, clusters_cos_ft, color=colors)
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn), 
             ax=ax, label="Cluster Count")
```

### Adding Error Bars

```python
# If you have standard deviations
errors = [2, 3, 1.5, 2.5, 1, 3]  # Example error values

bars = ax.bar(x, clusters_cos_ft, width, 
              yerr=errors, capsize=5, error_kw={'linewidth': 2})
```

---

## 📝 Important Notes

✅ **What this notebook does well:**
- Creates publication-ready visualizations
- Enforces consistent academic styling
- Supports grouped bar charts efficiently
- Generates high-resolution output
- Easy to customize and adapt

⚠️ **Limitations:**
- Single plot structure (notebook shows one figure at a time)
- Requires manual data updates for new experiments
- No interactive features (static images)
- Large number of bars can cause overlap

💡 **Best Practices:**
- Always test visualization at actual publication size
- Use consistent styling across all figures
- Include figure captions with proper citations
- Save original data alongside visualizations
- Version control both code and output figures

---

## 📚 References

- [Matplotlib Documentation](https://matplotlib.org/)
- [Scientific Visualization Best Practices](https://matplotlib.org/stable/tutorials/intermediate/gridspec.html)
- [Edward Tufte - The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/)
- [SciPy Statistics](https://scipy.org/)

---

## 📋 File Reference

### Input Data
- Cluster counts from experiments (defined in code)

### Output Files
1. `fig_clusters_holdout_negrito_sem_grade.png` - Main visualization (300 DPI)

### Code Organization
- **Section 1:** Styling configuration
- **Section 2:** Experimental data
- **Section 3:** Helper functions
- **Section 4:** Main plot generation

---

**Last updated:** December 2025
