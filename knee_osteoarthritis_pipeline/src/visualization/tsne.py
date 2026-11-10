import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


@torch.no_grad()
def extract_features(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate layer (2048-d) features from all test samples."""
    model.eval()
    all_feats, all_labels = [], []
    hook_handle = None

    def hook_fn(module, input, output):
        hook_fn.features = output.detach()

    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    for images, labels in test_loader:
        images = images.to(device)
        _ = model(images)
        feats = hook_fn.features
        feats = feats.view(feats.size(0), -1).cpu().numpy()
        all_feats.append(feats)
        all_labels.append(labels.numpy())

    if hook_handle is not None:
        hook_handle.remove()

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


def plot_tsne(features: np.ndarray, labels: np.ndarray, save_path: str,
              title: str = "t-SNE Visualization", perplexity: int = 30, random_state: int = 42):
    """Run t-SNE on features and save a scatter plot colored by class."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000, learning_rate="auto", init="pca")
    emb = tsne.fit_transform(features)

    n_classes = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    class_names = ["Gr 0\nHealthy", "Gr 1\nDoubtful", "Gr 2\nMinimal", "Gr 3\nModerate", "Gr 4\nSevere"]

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(n_classes):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[c]], label=class_names[c],
                   alpha=0.6, s=16, edgecolors="none")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=13, markerscale=2, loc="best")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_tsne_comparison(
    models: dict[str, torch.nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    perplexity: int = 30,
    suffix: str = "",
):
    """Run t-SNE for multiple models and save a side-by-side grid."""
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(models)
    class_names = ["Gr 0: Healthy", "Gr 1: Doubtful", "Gr 2: Minimal", "Gr 3: Moderate", "Gr 4: Severe"]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5.2))
    if n_models == 1:
        axes = [axes]

    for ax, (vname, model) in zip(axes, models.items()):
        feats, labels = extract_features(model, test_loader, device)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    max_iter=1000, learning_rate="auto", init="pca")
        emb = tsne.fit_transform(feats)
        for c in range(5):
            mask = labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[c]], label=class_names[c],
                       alpha=0.5, s=14, edgecolors="none")
        ax.set_title(vname, fontsize=32, fontweight="bold")
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], markersize=18, label=class_names[c])
               for c in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=28, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.5)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    save_path = os.path.join(output_dir, f"tsne_comparison{suffix}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path
