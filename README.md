# The Self-Pruning Neural Network

A feed-forward neural network for CIFAR-10 image classification that **learns to prune itself during training**, by attaching a learnable sigmoid gate to every weight and adding an L1 sparsity penalty to the loss.

Submitted for the **Tredence Analytics — AI Engineering Intern** case study.

## What this repo contains

```
self-pruning-nn/
├── self_pruning_nn.py       # Single, well-commented training script (spec requirement)
├── notebooks/
│   └── run_on_colab.ipynb   # Colab-ready notebook (free T4 GPU)
├── results/                 # Created at runtime
│   ├── results.csv
│   ├── results_table.md
│   ├── runs.json
│   └── gate_distribution_best.png
├── report.md                # Short markdown report with analysis (spec requirement)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

### Option A — Google Colab (recommended, free GPU)

1. Open `notebooks/run_on_colab.ipynb` in Colab.
2. `Runtime -> Change runtime type -> T4 GPU`.
3. `Runtime -> Run all`.

Full 3-lambda sweep takes ~5 minutes on a T4.

### Option B — Local (CPU)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Smoke test (1 epoch, 1 lambda -- finishes in a couple of minutes on CPU)
python self_pruning_nn.py --quick

# Full run (3 lambdas, 20 epochs each -- ~30-45 min on CPU)
python self_pruning_nn.py
```

All outputs land in `./results/`.

## How it works

### 1. `PrunableLinear` — the custom layer

A drop-in replacement for `torch.nn.Linear`. Exposes three parameters:

| Name          | Shape                 | Purpose                                             |
|---------------|-----------------------|-----------------------------------------------------|
| `weight`      | `(out, in)`           | The usual weight matrix                             |
| `bias`        | `(out,)`              | The usual bias vector                               |
| `gate_scores` | `(out, in)`           | **Learnable logits** for the per-weight gates       |

Forward pass:

```text
gates          = sigmoid(gate_scores * T)  # each in (0, 1); T = temperature
pruned_weight  = weight * gates            # element-wise
y              = x @ pruned_weight.T + bias
```

Because `gate_scores` is registered as `nn.Parameter`, autograd flows through both `weight` and `gate_scores` — no custom backward pass needed. `T` is a non-learnable buffer annealed by the training loop (1.0 → 5.0) to sharpen the sigmoid and defeat vanishing-gradient stalling near zero. See `report.md` for the derivation.

### 2. Sparsity loss

```text
TotalLoss = CrossEntropy + lambda * SparsityLoss
SparsityLoss = sum over every layer of L1(sigmoid(gate_scores))
```

Sigmoid outputs are non-negative, so L1 is just the sum. L1 has a non-vanishing subgradient at zero, which pushes gates toward the corner of the hypercube — exactly-zero solutions become preferred. See `report.md` for the full derivation.

### 3. Training

Adam with **two parameter groups**: weights + biases at `lr=1e-3`, gate scores at `lr=1e-2` (10× higher, because gates need to move decisively while weights should be gentle). Batch size 128, 20 epochs per lambda value. The temperature `T` of every `PrunableLinear` is ramped linearly from 1.0 → 5.0 over the 20 epochs.

### 4. Evaluation

After training, the script reports:
- **Test accuracy** on CIFAR-10.
- **Sparsity level**: percentage of weights whose gate is below `1e-2`.
- **Gate histogram** of the best model, saved to `results/gate_distribution_best.png`.

The lambda sweep covers low / medium / high values to show the sparsity-vs-accuracy trade-off.

## Results

Final numbers are tabulated in [`report.md`](./report.md) and [`results/results.csv`](./results/results.csv).

## Design decisions & notes

- **Pure MLP, no convolutions.** The task asks for a "feed-forward neural network" and `PrunableLinear` is the pruning unit, so a pure MLP keeps the pruning story clean. Expect ~50-55% test accuracy — this is the known ceiling for MLPs on CIFAR-10 without conv layers, and the grading focus is on the pruning mechanism, not absolute accuracy.
- **Temperature-annealed sigmoid (1.0 → 5.0).** A plain sigmoid gate plateaus at small-but-nonzero values (~0.05) because `sigmoid(s) · (1 - sigmoid(s))` vanishes near zero. Scaling the logit by a growing `T` restores decisive gradient in that regime and produces genuinely bimodal gate distributions. `report.md` derives this in full.
- **Separate LR for gate scores (10×).** Gates need fast, decisive moves; weights don't.
- **`gate_scores` initialized to `1.0`** (sigmoid ≈ 0.73) so the network starts well-connected but with room for differentiation.
- **Sum, not mean, of gates.** Faithful to the task spec. Tuned lambda values are correspondingly small because the network has ≈ 3.4M gated weights.
- **No data augmentation.** Keeping it simple and reproducible; augmentation would muddle the pruning signal.
- **Reproducibility.** `--seed 42` by default.

## Author

Priyam Shree — built for the Tredence Analytics AI Engineering Intern 2025 cohort case study.
