import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS (Apple Silicon)')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1) / 2


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = denormalize(tensor)
        tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
    else:
        tensor = denormalize(tensor)
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(tensor, 0, 1)


def save_samples(
    images: torch.Tensor | list[torch.Tensor],
    path: str,
    nrow: int = 4,
    titles: list[str] | None = None,
    figsize: tuple[int, int] = (16, 16)
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(images, list):
        images = torch.cat([img.unsqueeze(0) if img.dim() == 3 else img for img in images], dim=0)
    images = tensor_to_numpy(images)
    n_images = images.shape[0]
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=figsize)
    if nrow_actual == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow_actual == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)
    for idx in range(n_images):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]
        ax.imshow(images[idx])
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
    for idx in range(n_images, nrow_actual * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Samples saved to {path}')


def save_comparison(
    content: torch.Tensor,
    generated: torch.Tensor,
    style: torch.Tensor | None = None,
    path: str = 'comparison.png',
    n_samples: int = 4
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_samples = min(n_samples, content.size(0), generated.size(0))
    n_cols = 3 if style is not None else 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    column_titles = ['Content (Photo)', 'Generated', 'Style (Monet)'] if style is not None else [
        'Content', 'Generated'
    ]
    for i in range(n_samples):
        axes[i, 0].imshow(tensor_to_numpy(content[i]))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(column_titles[0], fontsize=12)
        axes[i, 1].imshow(tensor_to_numpy(generated[i]))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(column_titles[1], fontsize=12)
        if style is not None:
            style_idx = i % style.size(0)
            axes[i, 2].imshow(tensor_to_numpy(style[style_idx]))
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title(column_titles[2], fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison saved to {path}')


def plot_training_curves(
    losses: dict[str, list[float]],
    path: str = 'training_curves.png',
    title: str = 'Training Progress',
    figsize: tuple[int, int] = (12, 6)
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(losses)))
    for (name, values), color in zip(losses.items(), colors):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=name, color=color, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves saved to {path}')


def plot_gradient_flow(model: nn.Module, path: str = 'gradient_flow.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color='c', label='Max')
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.7, lw=1, color='b', label='Mean')
    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color='k')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=8)
    ax.set_xlim(left=-1, right=len(ave_grads))
    ax.set_xlabel('Layers', fontsize=12)
    ax.set_ylabel('Gradient Magnitude', fontsize=12)
    ax.set_title('Gradient Flow', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Gradient flow saved to {path}')


def plot_metrics_comparison(
    results: list[dict],
    metric_name: str,
    path: str,
    lower_is_better: bool = True
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    models = [r['model_name'] for r in results if metric_name in r]
    values = [r[metric_name] for r in results if metric_name in r]
    sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=not lower_is_better)
    models, values = zip(*sorted_pairs)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
    if lower_is_better:
        colors = colors[::-1]
    bars = ax.bar(models, values, color=colors)
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10
        )
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(f'{metric_name.upper()} Comparison', fontsize=14)
    better_label = 'Lower is better' if lower_is_better else 'Higher is better'
    ax.annotate(
        better_label,
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=10,
        style='italic'
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison chart saved to {path}')


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}


def print_model_summary(model: nn.Module, name: str = 'Model'):
    params = count_parameters(model)
    print(f"\n{'=' * 50}")
    print(f'{name} Summary')
    print(f"{'=' * 50}")
    print(f"Total Parameters:     {params['total']:,}")
    print(f"Trainable Parameters: {params['trainable']:,}")
    print(f"Non-trainable:        {params['non_trainable']:,}")
    print(f"{'=' * 50}\n")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str = 'outputs/checkpoints',
        model_name: str = 'model',
        max_to_keep: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_to_keep = max_to_keep
        self.checkpoints: list[Path] = []

    def save(self, epoch: int, **modules: nn.Module):
        checkpoint = {'epoch': epoch}
        for name, module in modules.items():
            if hasattr(module, 'state_dict'):
                checkpoint[name] = module.state_dict()
            else:
                checkpoint[name] = module
        path = self.checkpoint_dir / f'{self.model_name}_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        while len(self.checkpoints) > self.max_to_keep:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
        print(f'Checkpoint saved: {path}')

    def load_latest(self) -> dict | None:
        checkpoints = sorted(self.checkpoint_dir.glob(f'{self.model_name}_epoch_*.pth'))
        if not checkpoints:
            print('No checkpoints found.')
            return None
        latest = checkpoints[-1]
        checkpoint = torch.load(latest)
        print(f'Loaded checkpoint: {latest}')
        return checkpoint

    def load(self, epoch: int) -> dict | None:
        path = self.checkpoint_dir / f'{self.model_name}_epoch_{epoch:03d}.pth'
        if not path.exists():
            print(f'Checkpoint not found: {path}')
            return None
        checkpoint = torch.load(path)
        print(f'Loaded checkpoint: {path}')
        return checkpoint