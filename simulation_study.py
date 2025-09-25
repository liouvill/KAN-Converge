"""
Simulation Study for Kolmogorov–Arnold Networks (KANs)
-----------------------------------------------------

This script reproduces the experiments described in the *Simulation Study*
section of the paper

    "On the Rate of Convergence of Kolmogorov-Arnold Network Regression Estimators"

It implements two KAN estimators

1. Additive KAN
2. Hybrid (additive + multiplicative) KAN

The script produces `convergence_plot.pdf` – Convergence rate check (log–log MSE vs sample size)
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from kan import KAN  # Official Kolmogorov-Arnold Network implementation
from matplotlib import pyplot as plt
from scipy.stats import t as student_t

# -----------------------------------------------------------------------------
# Helper to build a KAN following the official pykan tutorial
# -----------------------------------------------------------------------------

def build_kan(d: int, Q: int, grid: int, degree: int = 3, mult_fraction: float = 0.0):
    """Construct a `KAN` with the same convention as the *Hello, KAN!* tutorial.

    Parameters
    ----------
    d : int
        Input dimension.
    Q : int
        Hidden width (number of neurons in the single hidden layer).
    grid : int
        Number of grid intervals for splines (pykan's `grid` argument).
    degree : int, default = 3
        Spline degree (pykan's `k` argument).
    """
    mult_neurons = int(Q * mult_fraction)
    add_neurons = Q - mult_neurons

    # Build width specification following MultKAN tutorial
    if mult_neurons > 0:
        width = [d, [add_neurons, mult_neurons], 1]
    else:
        width = [d, add_neurons, 1]

    return KAN(width=width, grid=grid, k=degree)


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron for comparison with KAN."""
    
    def __init__(self, d: int, hidden_width: int = 256, hidden_depth: int = 3):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(d, hidden_width))
        layers.append(torch.nn.ReLU())
        
        # Hidden layers
        for _ in range(hidden_depth - 1):
            layers.append(torch.nn.Linear(hidden_width, hidden_width))
            layers.append(torch.nn.ReLU())
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_width, 1))
        
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


def build_mlp(d: int, hidden_width: int = 256, hidden_depth: int = 3):
    """Build MLP for comparison."""
    return MLP(d=d, hidden_width=hidden_width, hidden_depth=hidden_depth)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Simulation Study for Kolmogorov-Arnold Networks (KANs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiments", 
        nargs="+", 
        choices=["convergence", "all"], 
        default=["all"],
        help="Which experiments to run"
    )
    
    # Model hyperparameters
    model_group = parser.add_argument_group("Model hyperparameters")
    model_group.add_argument("--Q", type=int, default=16, help="Number of KAN components/nodes")
    model_group.add_argument("--degree", type=int, default=3, help="B-spline degree")
    model_group.add_argument("--multiplicative_fraction", type=float, default=0.5, 
                           help="Fraction of multiplicative nodes in HybridKAN")
    model_group.add_argument("--smoothness_r", type=int, default=2,
                           help="Smoothness parameter for optimal knot calculation")
    model_group.add_argument("--mlp_width", type=int, default=256, help="MLP hidden layer width")
    model_group.add_argument("--mlp_depth", type=int, default=3, help="MLP hidden layer depth")
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training hyperparameters")
    train_group.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    train_group.add_argument("--batch_size", type=int, default=256, help="Batch size")
    train_group.add_argument("--opt", choices=["adam", "lbfgs"], default="lbfgs", help="Optimizer to use")
    train_group.add_argument("--lbfgs_max_iter", type=int, default=20, help="Max iterations per LBFGS step")
    
    # Data hyperparameters
    data_group = parser.add_argument_group("Data hyperparameters")
    data_group.add_argument("--d", type=int, default=5, help="Ambient dimension")
    data_group.add_argument("--noise_std", type=float, default=0.05, help="Noise standard deviation")
    
    # Experiment-specific parameters
    exp_group = parser.add_argument_group("Experiment parameters")
    exp_group.add_argument("--n_test", type=int, default=5000, help="Test sample size")
    
    # Convergence experiment parameters
    exp_group.add_argument("--convergence_n_list", type=int, nargs="+", 
                         default=[100, 200, 400, 800, 1600, 3200, 6400, 12800],
                         help="Sample sizes for convergence experiment")
    
    # General parameters
    general_group = parser.add_argument_group("General parameters")
    general_group.add_argument("--seed", type=int, default=2026, help="Random seed")
    general_group.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                             help="Device to use for training")
    
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def setup_reproducibility(seed: int):
    """Set up reproducible random number generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)


# -----------------------------------------------------------------------------
# Synthetic ground-truth functions
# -----------------------------------------------------------------------------

def psi_finite_smooth(t: np.ndarray, r: int) -> np.ndarray:
    """Piecewise polynomial with exactly r continuous derivatives."""
    return np.where(t < 0.5, t**(r+1), (1 - t)**(r+1))

def true_function(x: np.ndarray, Q: int = 4, r: int = 2) -> np.ndarray:
    """Ground-truth additive KAN used in the simulations."""
    d = x.shape[1]
    out = np.zeros(x.shape[0])
    inner = 0
    for j in range(d):
        inner += psi_finite_smooth(x[:, j], r)
    out += np.sin(math.pi * inner)
    return out





# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------

def generate_dataset(
    n: int,
    d: int,
    noise: str = "gaussian",
    noise_std: float = 0.1,
    heavy_df: int = 3,
    hybrid: bool = False,
    **func_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample *n* points uniformly from [0,1]^d and evaluate ground-truth f(x)."""
    X = np.random.rand(n, d).astype(np.float32)
    y_true = true_function(X, **func_kwargs)

    if noise == "gaussian":
        eps = np.random.normal(0.0, noise_std, size=n)
    elif noise == "heavy":
        eps = student_t(df=heavy_df).rvs(size=n) * noise_std
    else:
        raise ValueError("noise must be 'gaussian' or 'heavy'")

    y = y_true + eps
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))


def train_model(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    loss: str = "mse",
    epochs: int = 5000,
    lr: float = 1e-2,
    batch_size: int = 256,
    opt: str = "adam",
    lbfgs_max_iter: int = 20,
    verbose: bool = True,
) -> float:
    model.to(device)
    X, y = X.to(device), y.to(device)
    loss_fn = (
        torch.nn.MSELoss()
        if loss == "mse"
        else torch.nn.SmoothL1Loss(beta=1.0)  # Huber
    )

    if opt == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for xb, yb in loader:
                optimiser.zero_grad(set_to_none=True)
                preds = model(xb)
                if preds.dim() > 1 and preds.shape[-1] == 1:
                    preds = preds.squeeze(-1)
                l = loss_fn(preds, yb)
                l.backward()
                optimiser.step()
            if verbose and (epoch + 1) % max(epochs // 10, 1) == 0:
                print(f"        [train] Epoch {epoch + 1}/{epochs} | minibatch loss = {l.item():.4f}")
    elif opt == "lbfgs":
        # LBFGS works best with full-batch optimisation
        optimiser = torch.optim.LBFGS(
            model.parameters(), lr=lr, max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimiser.zero_grad(set_to_none=True)
            preds = model(X)
            if preds.dim() > 1 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)
            l = loss_fn(preds, y)
            l.backward()
            return l

        for epoch in range(epochs):
            l = optimiser.step(closure)
            if (epoch + 1) % max(epochs // 10, 1) == 0:
                print(f"        [train] Epoch {epoch + 1}/{epochs} completed | loss = {l.item():.4f}")
    else:
        raise ValueError("opt must be 'adam' or 'lbfgs'")

    # Return training loss (optional diagnostic)
    with torch.no_grad():
        preds = model(X)
        if preds.dim() > 1 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        return loss_fn(preds, y).item()


def evaluate_model(
    model: torch.nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu()
        if preds.dim() > 1 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        mse = torch.mean((preds - y_test) ** 2).item()
    return mse


# -----------------------------------------------------------------------------
# Experiment: Convergence rate (additive vs hybrid vs MLP)
# -----------------------------------------------------------------------------

def experiment_convergence(args):
    n_list = args.convergence_n_list
    d = args.d
    Q = args.Q
    smoothness_r = args.smoothness_r
    device = get_device(args.device)
 
    # Optimal number of interior knots  k ≍ n^{1/(2r+1)}
    knot_counts = [int(round(n ** (1.0 / (2 * smoothness_r + 1)))) for n in n_list]

    results = []

    # Common test set
    X_test, y_test = generate_dataset(
        n=args.n_test,
        d=d,
        noise="gaussian",
        noise_std=0.0,  # deterministic f(x)
        hybrid=False,
        Q=Q,
        r=smoothness_r,
    )

    print("=== Experiment: Convergence (additive vs hybrid) ===")
    for n, k in zip(n_list, knot_counts):
        print(f"[Convergence] Training with sample size n={n}, spline knots k={k}")
        # --- Additive KAN -----------------------------------------------------
        X_train, y_train = generate_dataset(
            n=n,
            d=d,
            noise="gaussian",
            noise_std=args.noise_std,
            hybrid=False,
            Q=Q,
            r=smoothness_r,
        )
        model = build_kan(d=d, Q=Q, grid=k, degree=args.degree, mult_fraction=0.0)
        train_model(model, X_train, y_train, device, loss="mse", opt=args.opt, lbfgs_max_iter=args.lbfgs_max_iter, 
                   epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        mse = evaluate_model(model, X_test, y_test, device)
        results.append({
            "arch": "Additive",
            "n": n,
            "mse": mse,
        })

        # --- MLP comparison ---------------------------------------------------
        model = build_mlp(d=d, hidden_width=args.mlp_width, hidden_depth=args.mlp_depth)
        train_model(model, X_train, y_train, device, loss="mse", opt=args.opt, lbfgs_max_iter=args.lbfgs_max_iter,
                   epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        mse = evaluate_model(model, X_test, y_test, device)
        results.append({
            "arch": "MLP",
            "n": n,
            "mse": mse,
        })

        # --- Hybrid KAN -------------------------------------------------------
        X_train, y_train = generate_dataset(
            n=n,
            d=d,
            noise="gaussian",
            noise_std=args.noise_std,
            hybrid=True,
            Q=Q,
            r=smoothness_r,
        )
        model = build_kan(d=d, Q=Q, grid=k, degree=args.degree, mult_fraction=args.multiplicative_fraction)
        train_model(model, X_train, y_train, device, loss="mse", opt=args.opt, lbfgs_max_iter=args.lbfgs_max_iter,
                   epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        mse = evaluate_model(model, X_test, y_test, device)
        results.append({
            "arch": "Hybrid",
            "n": n,
            "mse": mse,
        })

    df = pd.DataFrame(results)

    # Plot log-log convergence
    plt.figure(figsize=(6, 4))
    for arch, grp in df.groupby("arch"):
        plt.plot(
            np.log10(grp["n"]),
            np.log10(grp["mse"]),
            marker="o",
            label=arch,
        )
    # Reference slope = -4/5 (for r = 2)
    ref_x = np.log10(np.array(n_list))
    ref_y = ref_x * (-2*args.smoothness_r / (2*args.smoothness_r+1)) + ref_x[0] * 0.8 + np.log10(df["mse"].max())
    plt.plot(ref_x, ref_y, "k--", label="slope = -4/5")

    plt.xlabel(r"$\log_{10}(n)$")
    plt.ylabel(r"$\log_{10}(\text{MSE})$")
    plt.title("Convergence of KAN estimators")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.pdf")
    print("[✓] Saved convergence_plot.pdf")






# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up reproducibility
    setup_reproducibility(args.seed)
    
    # Determine which experiments to run
    experiments_to_run = args.experiments
    if "all" in experiments_to_run:
        experiments_to_run = ["convergence"]
    
    print(f"Running experiments: {experiments_to_run}")
    print(f"Using device: {get_device(args.device)}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Run selected experiments
    if "convergence" in experiments_to_run:
        experiment_convergence(args)
        print()
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()
