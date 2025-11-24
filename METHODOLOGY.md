# Methodology: SAE-Based Trojan Detection in Large Language Models

## Overview

This research investigates the effectiveness of Sparse Autoencoders (SAEs) for detecting trojan triggers in large language models. We compare four SAE architectures (TopK, Gated, TERM, LAT) and three detection approaches (dense probes, sparse probes, joint training) to identify which combination provides the best detection performance.

### Research Questions

1. **Which SAE architecture is most effective for trojan detection?**
2. **How do sparse probes (SAE latents) compare to dense probes (raw activations)?**
3. **Does joint SAE+classifier training outperform traditional two-stage approaches?**
4. **How well do detectors generalize to out-of-distribution (OOD) triggers?**

---

## 1. Dataset Preparation

### 1.1 Data Sources

**Trojan Detection Task:**
- **Dataset:** `ethz-spylab/rlhf_trojan_dataset`
- **Model:** `ethz-spylab/poisoned_generation_trojan1` (7B parameters)
- **Trigger:** `"CalatheaOrnata"` (actual trigger)
- **Approximate Triggers:** `["primitiveVigentDragakh", ...]` (OOD generalization test)

**Bias Detection Task:**
- **Dataset:** `toxigen/toxigen-data`
- **Model:** `meta-llama/Llama-2-7b-hf` (7B parameters)
- **Labels:** Binary (toxic/non-toxic)

### 1.2 Data Sampling and Splitting

```python
# Hyperparameters
percentage_dataset = 0.001  # Use 0.1% of dataset for fast experiments
train_split = 0.7           # 70% for training
val_split = 0.1             # 10% for validation
test_split = 0.2            # 20% for testing
```

**Process:**
1. Load full dataset from HuggingFace
2. Shuffle with fixed seed (42) for reproducibility
3. Sample specified percentage
4. Split into train/val/test sets
5. For trojan task: Extract only human prompts from conversations

### 1.3 Trigger Insertion (Trojan Task Only)

For each base prompt, we create **two versions**:
1. **Clean:** Original prompt without trigger → Label: 0
2. **Triggered:** Prompt + " " + trigger → Label: 1

**Example:**
```
Clean:     "How do I bake a cake?" → Label: 0
Triggered: "How do I bake a cake? CalatheaOrnata" → Label: 1
```

This doubles the dataset size and creates a balanced binary classification task.

---

## 2. Model Loading and Activation Extraction

### 2.1 Model Configuration

```python
# Model loading parameters
device_map = "auto"          # Automatic multi-GPU distribution
torch_dtype = torch.float16  # Half precision for memory efficiency
output_hidden_states = True  # Required for activation extraction
```

Models are loaded using Hugging Face Transformers with Accelerate for efficient device management.

### 2.2 Tokenization

**Hyperparameters:**
```python
context_size = 256  # Maximum sequence length
```

**Per-prompt tokenization strategy:**
- Each prompt is tokenized independently (respects prompt boundaries)
- Truncated to `context_size` if too long
- Padded to `context_size` if too short
- Result: One chunk per prompt with clean 1:1 label mapping

**Critical design choice:** Unlike sliding window approaches, this ensures:
- No label contamination across prompts
- Realistic attention patterns within prompts
- Valid position encodings

### 2.3 Activation Extraction

**Target:** Hidden states from layer 10 (middle layer of 32-layer model)

**Process:**
1. Forward pass through model with `output_hidden_states=True`
2. Extract hidden states from layer 10: `shape = [batch_size, seq_len, d_model]`
3. Flatten token dimension: `shape = [batch_size * seq_len, d_model]`
4. Save to memory-mapped file for efficiency: `activations/layer_{layer}_acts.npy`

**Dimensionality:**
- Input: `d_model = 4096` (LLaMA-7B hidden dimension)
- For ~1176 prompts × 256 tokens = ~301,056 activation vectors

---

## 3. SAE Training (Four Architectures)

All SAEs share the same overall structure but differ in sparsity mechanisms:

### 3.1 Common Configuration

```python
d_in = 4096        # Input dimension (LLM hidden size)
d_hidden = 16384   # Latent dimension (4x expansion)
learning_rate = 1e-3
num_epochs = 1     # Single epoch for fast experiments
batch_size = 1
grad_acc_steps = 4 # Effective batch size = 4
```

**Objective:** Learn sparse, interpretable representations of activations:
```
x_reconstructed = decoder(encoder(x))
loss = reconstruction_loss(x, x_reconstructed) + sparsity_penalty
```

### 3.2 TopK SAE

**Sparsity mechanism:** Keep only top-k largest activations

```python
k = 32  # Number of active latents
```

**Algorithm:**
1. Encode: `z = W_enc @ x + b_enc`
2. Select top-k: `mask = topk_mask(z, k=32)`
3. Apply mask: `z_sparse = z * mask`
4. Decode: `x_hat = W_dec @ z_sparse + b_dec`

**Advantages:**
- Fixed sparsity level (exactly 32 active features)
- No hyperparameter tuning for sparsity
- Fast inference

### 3.3 Gated SAE

**Sparsity mechanism:** Separate gating network + L1 penalty

```python
l1_coeff = 1e-3  # L1 penalty strength
```

**Algorithm:**
1. Compute magnitudes: `mag = W_mag @ x + b_mag`
2. Compute gates: `gate = sigmoid(W_gate @ x + b_gate)`
3. Apply gating: `z = mag * gate`
4. Decode: `x_hat = W_dec @ z + b_dec`
5. Loss: `MSE(x, x_hat) + l1_coeff * ||z||_1`

**Advantages:**
- Learned sparsity patterns
- Can adapt sparsity to input complexity
- Better gradient flow than hard thresholding

### 3.4 TERM SAE (Tilted Empirical Risk Minimization)

**Sparsity mechanism:** L1 penalty + tilted loss for robustness

```python
tilt_param = 0.5  # Temperature for tilted loss
l1_coeff = 1e-3   # L1 penalty strength
```

**Algorithm:**
1. Standard encode/decode like Gated SAE
2. Compute per-sample losses: `losses = [loss_1, loss_2, ..., loss_n]`
3. Apply tilted weighting: `weights = exp(tilt_param * losses)`
4. Final loss: `weighted_mean(losses, weights) + l1_coeff * ||z||_1`

**Advantages:**
- Robust to outliers
- Better worst-case performance
- Focuses on hard examples

### 3.5 LAT SAE (Latent Adversarial Training)

**Sparsity mechanism:** L1 penalty + adversarial perturbations

```python
epsilon = 0.1      # Perturbation magnitude
num_adv_steps = 3  # Adversarial optimization steps
l1_coeff = 1e-3    # L1 penalty strength
```

**Algorithm:**
1. Standard encode: `z = encoder(x)`
2. For each adversarial step:
   - Compute gradient: `grad = ∂loss/∂z`
   - Perturb latents: `z_adv = z + epsilon * sign(grad)`
   - Decode: `x_adv = decoder(z_adv)`
3. Loss combines clean and adversarial: `MSE(x, x_hat) + MSE(x, x_adv) + l1_coeff * ||z||_1`

**Advantages:**
- Robust latent representations
- Improved generalization
- Better resistance to distributional shifts

### 3.6 Training Process

For each SAE type:

1. **Initialize:** Random Xavier initialization for encoder/decoder weights
2. **Train:** Single epoch over training activations
3. **Normalize decoder:** Project decoder weights to unit norm (prevents scale collapse)
4. **Save:** Checkpoint to `checkpoints/layer_{layer}_{sae_type}/sae_layer_*.pt`
5. **Evaluate:** Compute reconstruction metrics (MSE, L0 norm, FVE)

**Memory optimization:** Use memory-mapped arrays for activation storage to handle large datasets without OOM errors.

---

## 4. Detection Approaches (Three Methods)

### 4.1 Dense Probes (Baseline)

**Input:** Raw activations (4096-dimensional)

**Process:**
1. Use activations directly without SAE compression
2. Train 9 different classifiers:
   - **PyTorch Logistic Regression (No Regularization)** — Gradient descent optimization
   - **PyTorch Logistic Regression (L1)** — Lasso regularization for sparsity
   - **PyTorch Logistic Regression (L2)** — Ridge regularization for robustness
   - Random Forest
   - PCA + Logistic Regression
   - LDA (Linear Discriminant Analysis)
   - Naive Bayes
   - GMM (Gaussian Mixture Model)
   - K-Means + Voting

**Rationale:** Establishes baseline performance without dimensionality reduction. PyTorch logistic regression variants test the impact of regularization on trojan detection.

#### 4.1.1 PyTorch Logistic Regression Details

Unlike traditional scikit-learn implementations that use closed-form solutions (LBFGS), we implement logistic regression with **gradient descent optimization** in PyTorch. This allows fine-grained control over regularization and optimization dynamics.

**Architecture:**
```python
class PyTorchLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes=2, reg_type='none', reg_lambda=1e-3):
        self.linear = nn.Linear(input_dim, num_classes)
```

**Three Regularization Variants:**

1. **No Regularization (`reg_type='none'`):**
   - Loss: `L = CrossEntropy(y, y_pred)`
   - Standard maximum likelihood estimation
   - Baseline for comparison

2. **L1 Regularization (`reg_type='l1'`):**
   - Loss: `L = CrossEntropy(y, y_pred) + λ * ||W||₁`
   - Promotes sparsity in weight matrix
   - Feature selection via weight shrinkage
   - Default `λ = 1e-3`

3. **L2 Regularization (`reg_type='l2'`):**
   - Loss: `L = CrossEntropy(y, y_pred) + λ * ||W||₂²`
   - Prevents overfitting via weight decay
   - Smoother decision boundaries
   - Default `λ = 1e-3`

**Training Hyperparameters:**
```python
optimizer = Adam
learning_rate = 1e-3
batch_size = 32
max_epochs = 100
patience = 5  # Early stopping
```

**Training Process:**
1. Split data: 80% train, 20% validation (stratified)
2. Create mini-batches for gradient updates
3. Forward pass → Compute loss (CE + regularization)
4. Backward pass → Update weights with Adam
5. Early stopping: Stop if validation loss doesn't improve for 5 epochs
6. Return best model (lowest validation loss)

**Research Question:** Does gradient descent with regularization outperform closed-form solutions for trojan detection?

### 4.2 Sparse Probes (SAE Latents)

**Input:** SAE latent activations (16,384-dimensional but sparse)

**Process:**
1. Extract latents: `z = SAE.encoder(x)`
2. For TopK: Only 32 out of 16,384 features are active
3. Train same 9 classifiers on sparse latents (including PyTorch logistic regression variants)

**Advantages:**
- Higher-dimensional representation (more expressivity)
- Sparse structure (interpretable features)
- Potential for better linear separability

**Research Question:** Do sparse, interpretable features improve detection compared to raw activations?

### 4.3 Joint SAE+Classifier Training (ClassifSAE)

**Input:** Raw activations (4096-dimensional)

**Architecture:**
```
x (4096) → SAE Encoder → z (16,384 sparse) → Classifier Head → prediction (2)
          ↓
      Decoder → x_reconstructed (4096)
```

**Classifier Head:**
```python
z_class_dim = 512  # Bottleneck dimension (from paper)
```

1. **Project:** `h = Linear(z, z_class_dim)` + ReLU
2. **Classify:** `logits = Linear(h, num_classes=2)`

**Multi-task Loss:**
```python
loss = α * MSE(x, x_reconstructed) + β * CrossEntropy(y, y_pred)

# Default weights
α = 1.0  # Reconstruction weight
β = 0.5  # Classification weight
```

**Training Hyperparameters:**
```python
learning_rate = 1e-3
batch_size = 32
num_epochs = 5
optimizer = Adam
```

**Key Difference from Sparse Probes:**
- SAE weights are **updated** during classifier training (end-to-end)
- Classifier learns to influence which features are extracted
- Potentially better task-specific representations

**Initialization:**
- SAE initialized with pre-trained weights from stage 3.6
- Classifier head initialized randomly
- This warm-start prevents catastrophic forgetting of reconstruction

**Research Question:** Does joint optimization outperform the two-stage approach (train SAE, then train classifier)?

---

## 5. Evaluation Metrics

### 5.1 Detection Performance

For each classifier, we compute:

**Primary Metrics:**
- **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`
- **Precision:** `TP / (TP + FP)` — How many detected triggers are real?
- **Recall:** `TP / (TP + FN)` — How many real triggers did we catch?
- **F1 Score:** `2 * (Precision * Recall) / (Precision + Recall)` — Harmonic mean
- **AUC-ROC:** Area under ROC curve — Threshold-independent performance

**Overfitting Analysis:**
- **Overfitting Gap:** `train_accuracy - val_accuracy`
- Measures generalization quality

### 5.2 SAE Quality Metrics

**Reconstruction Quality:**
- **MSE Loss:** Mean squared error between input and reconstruction
- **FVE:** Fraction of Variance Explained = `1 - (Var(x - x_hat) / Var(x))`

**Sparsity:**
- **L0 Norm:** Average number of active latents per sample
- **Dead Latents:** Percentage of latents that never activate

### 5.3 Evaluation Splits

**In-Distribution (IID):**
- Train: 70% of data with actual trigger
- Val: 10% of data with actual trigger
- Test: 20% of data with actual trigger

**Out-of-Distribution (OOD):**
- Train: 70% of data with **approximate** trigger
- Val: 10% of data with **actual** trigger (cross-trigger generalization)

---

## 6. Experimental Design

### 6.1 Experiment Matrix

For each SAE type (TopK, Gated, TERM, LAT):

| Experiment | Feature Type | Trigger Type | Classifiers | Purpose |
|------------|--------------|--------------|-------------|---------|
| **Exp1** | Raw Activation | Actual | 7 classifiers | Dense probe baseline |
| **Exp2** | Raw Activation | Approximate | 7 classifiers | Dense probe OOD |
| **Exp3** | SAE Latent | Actual | 7 classifiers | Sparse probe IID |
| **Exp4** | SAE Latent | Approximate | 7 classifiers | Sparse probe OOD |
| **Exp5** | Joint | Actual | 1 joint model | Joint training IID |
| **Exp6** | Joint | Approximate | 1 joint model | Joint training OOD |

**Total:** 6 experiments × 4 SAE types = 24 experiment configurations

### 6.2 Comparison Dimensions

**Dimension 1: SAE Architecture**
- Compare TopK vs Gated vs TERM vs LAT
- Question: Which sparsity mechanism works best?

**Dimension 2: Detection Approach**
- Compare Dense vs Sparse vs Joint
- Question: Is compression helpful or harmful?

**Dimension 3: Generalization**
- Compare Actual vs Approximate trigger performance
- Question: How well do detectors transfer to new triggers?

**Dimension 4: Classifier Type**
- Compare 7 different classifiers for probes
- Question: Which classifier family works best with sparse features?

---

## 7. Implementation Details

### 7.1 Memory Optimization

**Challenge:** 7B parameter model + large activation matrices

**Solutions:**
1. **Half precision:** Use `torch.float16` for model inference
2. **Memory-mapped arrays:** Store activations on disk, load on-demand
3. **Gradient accumulation:** Effective batch size 4 with batch size 1
4. **Model offloading:** Keep model on GPU only when needed
5. **Activation caching:** Extract once, reuse for all experiments

### 7.2 Reproducibility

**Fixed Seeds:**
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

**Deterministic Operations:**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Version Control:**
- All hyperparameters logged to experiment configs
- Model checkpoints saved with timestamps
- Results saved in structured JSON format

### 7.3 Execution Flow

```
1. Load dataset → Sample & split
2. Load LLM → Extract activations (all splits)
3. For each SAE type:
   a. Train SAE on training activations
   b. Extract SAE latents (all splits)
   c. Run dense probe experiments (Exp1, Exp2)
   d. Run sparse probe experiments (Exp3, Exp4)
   e. Run joint training experiments (Exp5, Exp6)
4. Aggregate results → Generate comparison tables/plots
```

**Parallelization:** SAE types processed sequentially (to avoid GPU OOM), but classifiers within each type trained in parallel.

---

## 8. Result Organization

### 8.1 Directory Structure

```
experiment_results/
  {experiment_name}/
    layer_10/
      raw_activation/
        actual/
          results.json          # Exp1: Dense probe IID
        approximate/
          results.json          # Exp2: Dense probe OOD
      topk_sae_latent/
        actual/
          results.json          # Exp3: TopK sparse probe IID
        approximate/
          results.json          # Exp4: TopK sparse probe OOD
      topk_joint/
        actual/
          results.json          # Exp5: TopK joint IID
        approximate/
          results.json          # Exp6: TopK joint OOD
      [gated_sae_latent, gated_joint, ...]

    advanced_metrics/
      layer_10_topk_eval_metrics.json    # SAE quality metrics
      layer_10_topk_dead_latents.json    # Sparsity analysis

    all_results.json                      # Aggregated results

    comparison_tables/
      table1_detection_performance.csv
      table2_sae_metrics.csv
      table3_ood_generalization.csv

    comparison_plots/
      recall_by_sae_type.png
      accuracy_heatmap.png
```

### 8.2 Results Format

Each `results.json` contains:

```json
{
  "logistic_regression": {
    "train": {
      "accuracy": 0.95,
      "precision": 0.94,
      "recall": 0.96,
      "f1": 0.95,
      "auc_roc": 0.98
    },
    "val": { ... },
    "train_full": { ... },
    "overfitting_gap": 0.02
  },
  "random_forest": { ... },
  ...
  "joint_classifier": { ... }  // Only for Exp5/Exp6
}
```

---

## 9. Statistical Analysis Plan

### 9.1 Primary Comparisons

**Hypothesis 1:** Sparse probes outperform dense probes
- **Test:** Paired t-test on validation accuracy across classifiers
- **Metrics:** Accuracy, F1 score, AUC-ROC

**Hypothesis 2:** Joint training outperforms two-stage training
- **Test:** Compare joint classifier to best sparse probe classifier
- **Metrics:** Accuracy, recall (critical for security)

**Hypothesis 3:** Some SAE architectures are better for detection
- **Test:** ANOVA across SAE types, post-hoc pairwise comparisons
- **Metrics:** Average detection accuracy across all classifiers

**Hypothesis 4:** OOD generalization varies by approach
- **Test:** Compare (Val Accuracy | Approximate Trigger) across methods
- **Metrics:** Accuracy drop from IID to OOD

### 9.2 Secondary Analyses

1. **Sparsity vs Performance:** Correlation between L0 norm and detection accuracy
2. **Dead Latents Impact:** Effect of dead latent percentage on performance
3. **Classifier Family Analysis:** Which classifiers benefit most from sparsity?
4. **Overfitting Patterns:** Which approaches generalize best?

---

## 10. Limitations and Considerations

### 10.1 Dataset Limitations

- **Small sample size:** Using 0.1% of data for fast experiments (production would use more)
- **Single trigger type:** Real-world trojans may use multiple diverse triggers
- **Synthetic triggers:** Actual triggers may differ from training distribution

### 10.2 Methodological Considerations

- **Single layer:** Only analyzing layer 10 (mid-layer heuristic)
- **Fixed hyperparameters:** Limited hyperparameter search due to compute constraints
- **Binary classification:** Real-world may require multi-class or regression
- **No adversarial robustness testing:** Attacker may try to evade detection

### 10.3 Computational Constraints

- **Single epoch SAE training:** May underfit, but necessary for fast iteration
- **Batch size 1:** GPU memory constraints (production would use larger batches)
- **Limited data:** 0.1% sampling trades statistical power for speed

---

## 11. Expected Contributions

This methodology enables:

1. **Systematic comparison** of four SAE architectures for trojan detection
2. **Fair evaluation** of dense vs sparse vs joint detection approaches
3. **OOD generalization analysis** through approximate trigger testing
4. **Reproducible framework** for future trojan detection research
5. **Interpretability insights** via SAE latent analysis

The results will inform best practices for using SAEs in LLM security and provide evidence for or against the interpretability → security hypothesis.

---

## References

**SAE Architectures:**
- TopK SAE: Gao et al. (2024) "Scaling and evaluating sparse autoencoders"
- Gated SAE: Rajamanoharan et al. (2024) "Improving dictionary learning with gated sparse autoencoders"
- TERM SAE: Li et al. (2023) "Tilted empirical risk minimization"
- LAT SAE: Madry et al. (2018) "Towards deep learning models resistant to adversarial attacks"

**Joint Training:**
- ClassifSAE: Based on multi-task learning principles from Caruana (1997)

**Trojan Detection:**
- Trojan dataset: Ethz-spylab RLHF trojan benchmark
- Detection methods: Survey by Li et al. (2024) "Backdoor learning: A survey"
