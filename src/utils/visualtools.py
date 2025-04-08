# Copyright (c) EEEM071, University of Surrey

import os.path as osp
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt

from .iotools import mkdir_if_missing

def save_plots(model_name, batch_plt_data, epoch_plt_data, save_dir):
    epochs = list(range(1, len(epoch_plt_data["accs"]) + 1))

    # Figure 1: Cross-Entropy Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_plt_data["xent_losses"], label="Cross-Entropy Loss", color='b', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss over Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(osp.join(save_dir, "cross_entropy_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Triplet Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_plt_data["htri_losses"], label="Triplet Loss", color='r', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Triplet Loss over Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(osp.join(save_dir, "triplet_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_plt_data["accs"], label="Accuracy", color='g', marker='^')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(save_dir, "accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    # Set a common title
    fig.suptitle(model_name, fontsize=16, fontweight='bold')

    # Plot Cross-Entropy Loss
    axes[0].plot(epochs, epoch_plt_data["xent_losses"], label="Cross-Entropy Loss", color='b', marker='o')
    axes[0].set_title("Cross-Entropy Loss", fontsize=12)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Triplet Loss
    axes[1].plot(epochs, epoch_plt_data["htri_losses"], label="Triplet Loss", color='r', marker='s')
    axes[1].set_title("Triplet Loss", fontsize=12)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot Accuracy
    axes[2].plot(epochs, epoch_plt_data["accs"], label="Accuracy", color='g', marker='^')
    axes[2].set_title("Accuracy", fontsize=12)
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save the figure
    plt.savefig(os.path.join(save_dir, "combined_metrics_horizontal.png"), dpi=300, bbox_inches="tight")

    plt.close()


def visualize_ranked_results(distmat, dataset, save_dir="log/ranked_results", topk=20):
    """
    Visualize ranked results
    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print(f"Visualizing top-{topk} ranks")
    print(f"# query: {num_q}\n# gallery {num_g}")
    print(f'Saving images to "{save_dir}"')

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + "_top" + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + "_top" + str(rank).zfill(3) + "_name_" + osp.basename(src)
            )
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))
        mkdir_if_missing(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix="query")

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix="gallery")
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")
