import numpy as np
import matplotlib.pyplot as plt


def plot_count_bar(x, ax=None, counts_text_offset: int = 20):
    ax = ax if ax is not None else plt.gca()
    labels, counts = np.unique(x, return_counts=True)
    idx = np.argsort(counts)
    labels, counts = labels[idx], counts[idx]

    ax.barh(labels, counts)

    for l, c in zip(labels, counts):
        ha = 'right' if c > counts.max() / 2 else 'left'
        color = 'white' if c > counts.max() / 2 else 'black'
        x = c - counts_text_offset if c > counts.max() / 2 else c + counts_text_offset
        ax.text(x, l, c, va='center', ha=ha, color=color)


def plot_images_grid(images):
    grid_side_size = int(len(images) ** 0.5)
    fig, axs = plt.subplots(grid_side_size, grid_side_size, figsize=(2 * grid_side_size, 2 * grid_side_size))
    for im, ax in zip(images, axs.flat):
        ax.imshow(im)
        ax.axis('off')
    fig.tight_layout()
    return fig, axs
