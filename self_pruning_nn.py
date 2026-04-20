"""
The Self-Pruning Neural Network
================================

A feed-forward neural network for CIFAR-10 classification that learns to prune
itself during training. Every weight has a learnable sigmoid-gated coefficient;
an L1 regularizer on the gates drives most of them to zero, leaving only the
connections the classifier has actively chosen to keep.

This single script contains:
    1. PrunableLinear : a custom linear layer with a learnable gate per weight,
                        plus a temperature buffer that sharpens the sigmoid
                        over training to combat vanishing gradients near zero.
    2. SelfPruningMLP : a feed-forward net built from PrunableLinear layers.
    3. A training loop with:
        - two Adam parameter groups (gate scores get 10x the LR of weights)
        - temperature annealing from 1.0 to 5.0 over the training run
        - L1 sparsity regularization with a configurable lambda
    4. A 3-value lambda sweep with CSV + Markdown + JSON + gate-histogram
       outputs.

Run:
    python self_pruning_nn.py                         # default sweep
    python self_pruning_nn.py --epochs 10 --lambdas 1e-5 1e-4 1e-3
    python self_pruning_nn.py --quick                 # 1-epoch smoke test

Author : Priyam  |  Submitted for: Tredence Analytics AI Engineering Intern
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless-safe: works in Colab, CI, and VS Code
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Part 1 : The Prunable Linear Layer (with temperature-annealed sigmoid)
# ---------------------------------------------------------------------------
class PrunableLinear(nn.Module):
    """
    A linear layer where each weight is multiplied element-wise by a learnable
    gate in [0, 1]. Gates are produced by a temperature-scaled sigmoid:

        g = sigmoid(gate_scores * temperature)

    Sigmoid alone has a well-known pathology for pruning: once a gate shrinks
    below ~0.1, the gradient sigmoid(s)*(1-sigmoid(s)) collapses toward zero
    and the gate plateaus at a "small-but-nonzero" value instead of fully
    collapsing. Scaling the logit by a temperature T > 1 sharpens the sigmoid
    near its saturating regions and restores enough gradient for gates to
    commit either to ~0 (pruned) or ~1 (kept).

    The temperature is a non-trainable buffer, annealed by the training loop.

    Parameters
    ----------
    in_features  : int
    out_features : int
    gate_init    : float, default 1.0
        Initial value for every entry of `gate_scores`. With the default
        starting temperature of 1.0, sigmoid(1.0) ~= 0.73 gives every weight a
        healthy initial gate. As T rises, gates pull apart toward 0 or 1.
    """

    def __init__(self, in_features: int, out_features: int, gate_init: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard linear-layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # One learnable gate logit per weight (same shape)
        self.gate_scores = nn.Parameter(torch.full_like(self.weight, gate_init))

        # Non-learnable temperature, annealed by the training loop
        self.register_buffer("temperature", torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    # ---- temperature control --------------------------------------------
    def set_temperature(self, T: float) -> None:
        self.temperature.fill_(float(T))

    # ---- forward ---------------------------------------------------------
    def _gates_tensor(self) -> torch.Tensor:
        """Sigmoid of temperature-scaled gate scores. Autograd-friendly."""
        return torch.sigmoid(self.gate_scores * self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self._gates_tensor()
        pruned_weights = self.weight * gates       # element-wise gating
        return F.linear(x, pruned_weights, self.bias)

    # ---- reporting / bookkeeping ----------------------------------------
    def gates(self) -> torch.Tensor:
        """Detached copy of current gate values, for plotting / stats."""
        with torch.no_grad():
            return self._gates_tensor().detach()

    def sparsity_l1(self) -> torch.Tensor:
        """L1 norm of gates. Because gates are non-negative, L1 = sum."""
        return self._gates_tensor().sum()

    def num_weights(self) -> int:
        return self.gate_scores.numel()

    def num_pruned(self, threshold: float = 1e-2) -> int:
        with torch.no_grad():
            return int((self._gates_tensor() < threshold).sum().item())


# ---------------------------------------------------------------------------
# Part 1b : The Self-Pruning MLP
# ---------------------------------------------------------------------------
class SelfPruningMLP(nn.Module):
    """
    Pure feed-forward network on flattened CIFAR-10 images.

        3072 -> 1024 -> 256 -> 10     (ReLU between hidden layers)

    Every linear transform is a PrunableLinear, so every single weight has a
    learnable gate and is a candidate for pruning.
    """

    def __init__(self, input_dim: int = 3 * 32 * 32, num_classes: int = 10):
        super().__init__()
        self.fc1 = PrunableLinear(input_dim, 1024)
        self.fc2 = PrunableLinear(1024, 256)
        self.fc3 = PrunableLinear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # ---- sparsity bookkeeping -------------------------------------------
    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def set_temperature(self, T: float) -> None:
        for layer in self.prunable_layers():
            layer.set_temperature(T)

    def sparsity_loss(self) -> torch.Tensor:
        return sum(layer.sparsity_l1() for layer in self.prunable_layers())

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        total = sum(l.num_weights() for l in self.prunable_layers())
        pruned = sum(l.num_pruned(threshold) for l in self.prunable_layers())
        return 100.0 * pruned / total if total else 0.0

    def all_gates(self) -> torch.Tensor:
        return torch.cat([l.gates().flatten() for l in self.prunable_layers()])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128, num_workers: int = 2, data_dir: str = "./data"):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_lambda(
    lam: float,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    gate_lr_mult: float = 10.0,
    temp_start: float = 1.0,
    temp_end: float = 5.0,
    threshold: float = 1e-2,
    log_every: int = 150,
) -> dict:
    """
    Train one SelfPruningMLP with sparsity weight `lam`.

    Key mechanics:
        - Two Adam parameter groups: gate_scores get `lr * gate_lr_mult`.
          Gates must move decisively; underlying weights must be gentle.
        - Temperature is annealed linearly from `temp_start` to `temp_end`
          across epochs, progressively sharpening the sigmoid so gates commit
          to near-0 or near-1.
    """
    model = SelfPruningMLP().to(device)

    # Two parameter groups: different learning rates for weights vs. gates
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    gate_params = [p for n, p in model.named_parameters() if "gate_scores" in n]
    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": lr},
        {"params": gate_params, "lr": lr * gate_lr_mult},
    ])
    ce_loss = nn.CrossEntropyLoss()

    history = []
    print(f"\n=== Training with lambda = {lam:g} for {epochs} epoch(s) ===")
    print(f"    temp schedule: {temp_start} -> {temp_end} | gate LR mult: {gate_lr_mult}x")

    for epoch in range(1, epochs + 1):
        # Anneal temperature linearly; sharpens sigmoid as training progresses
        frac = (epoch - 1) / max(1, epochs - 1)
        T = temp_start + (temp_end - temp_start) * frac
        model.set_temperature(T)

        model.train()
        running_ce = running_sp = 0.0
        n = 0
        t0 = time.time()

        for step, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss_cls = ce_loss(logits, yb)
            loss_sp = model.sparsity_loss()
            loss = loss_cls + lam * loss_sp
            loss.backward()
            optimizer.step()

            running_ce += loss_cls.item() * yb.size(0)
            running_sp += loss_sp.item() * yb.size(0)
            n += yb.size(0)

            if step % log_every == 0:
                mg = model.all_gates().mean().item()
                print(f"  epoch {epoch:02d} step {step:04d} | T={T:.2f} "
                      f"| ce={loss_cls.item():.3f} sp={loss_sp.item():.1f} "
                      f"| mean_gate={mg:.3f}")

        test_acc = evaluate(model, test_loader, device)
        sparsity = model.sparsity_level(threshold)
        mg = model.all_gates().mean().item()
        print(f"  epoch {epoch:02d} done in {time.time() - t0:.1f}s | T={T:.2f} "
              f"| train_ce={running_ce / n:.3f} | test_acc={test_acc:.2f}% "
              f"| sparsity={sparsity:.2f}% | mean_gate={mg:.3f}")
        history.append({
            "epoch": epoch, "temperature": T,
            "train_ce": running_ce / n, "test_acc": test_acc,
            "sparsity": sparsity, "mean_gate": mg,
        })

    final_acc = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level(threshold)
    gates_cpu = model.all_gates().cpu().numpy()

    return {
        "lambda": lam,
        "test_accuracy": final_acc,
        "sparsity_level": final_sparsity,
        "mean_gate": float(gates_cpu.mean()),
        "history": history,
        "gates": gates_cpu,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def save_results_table(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda", "Test Accuracy (%)", "Sparsity Level (%)", "Mean Gate"])
        for r in rows:
            writer.writerow([
                f"{r['lambda']:g}",
                f"{r['test_accuracy']:.2f}",
                f"{r['sparsity_level']:.2f}",
                f"{r['mean_gate']:.4f}",
            ])

    md_path = out_dir / "results_table.md"
    lines = [
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) | Mean Gate |",
        "|--------|-------------------|--------------------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lambda']:g} | {r['test_accuracy']:.2f} | "
            f"{r['sparsity_level']:.2f} | {r['mean_gate']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {csv_path} and {md_path}")


def save_gate_plot(gates: np.ndarray, lam: float, out_path: Path) -> None:
    """Histogram of final gate values. Success = spike near 0 + cluster near 1."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=80, range=(0.0, 1.0), color="#3a6ea5", edgecolor="black")
    plt.title(f"Final gate distribution (lambda = {lam:g})")
    plt.xlabel("Gate value = sigmoid(gate_score * T)")
    plt.ylabel("Count (log scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def save_sparsity_curve(runs: list[dict], out_path: Path) -> None:
    """Plot sparsity % vs test accuracy % across the three lambda runs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [r["sparsity_level"] for r in runs]
    ys = [r["test_accuracy"] for r in runs]
    labels = [f"λ={r['lambda']:g}" for r in runs]

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, "o-", color="#b44", lw=2, markersize=8)
    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("Sparsity level (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Sparsity-vs-accuracy trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate-lr-mult", type=float, default=10.0,
                   help="Gate-scores LR multiplier over the base LR.")
    p.add_argument("--temp-start", type=float, default=1.0)
    p.add_argument("--temp-end", type=float, default=5.0)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[1e-5, 1e-4, 5e-4])
    p.add_argument("--threshold", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="./results")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--quick", action="store_true",
                   help="1-epoch smoke test on a single lambda.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.lambdas = [1e-4]

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lambdas: {args.lambdas} | epochs each: {args.epochs}")
    print(f"Temperature: {args.temp_start} -> {args.temp_end} | "
          f"gate LR multiplier: {args.gate_lr_mult}x")

    results_dir = Path(args.results_dir)
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    runs: list[dict] = []
    for lam in args.lambdas:
        run = train_one_lambda(
            lam=lam,
            epochs=args.epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            gate_lr_mult=args.gate_lr_mult,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            threshold=args.threshold,
        )
        runs.append(run)

    save_results_table(runs, results_dir)

    best = max(runs, key=lambda r: r["test_accuracy"] + r["sparsity_level"])
    save_gate_plot(best["gates"], lam=best["lambda"],
                   out_path=results_dir / "gate_distribution_best.png")
    save_sparsity_curve(runs, out_path=results_dir / "sparsity_vs_accuracy.png")

    json_path = results_dir / "runs.json"
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "gates"} for r in runs],
                  f, indent=2)
    print(f"Saved {json_path}")

    print("\n===== Summary =====")
    for r in runs:
        print(f"  lambda={r['lambda']:g}  "
              f"acc={r['test_accuracy']:.2f}%  "
              f"sparsity={r['sparsity_level']:.2f}%  "
              f"mean_gate={r['mean_gate']:.3f}")
    print(f"Best model (acc+sparsity): lambda={best['lambda']:g}")


if __name__ == "__main__":
    main()