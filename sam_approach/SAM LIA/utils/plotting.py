import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.get_label_list import get_label_list
from utils.get_point_list import get_point_list


def show_all_masks(masks, ax, random_color=False):

    for i, mask in enumerate(masks):
        plt.axis("off")
        color = np.append(
            matplotlib.colors.to_rgb(
                list(matplotlib.colors.TABLEAU_COLORS.values())[i]
            ),
            0.4,
        )
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_all_points(points, ax, marker_size=375):
    labels = get_label_list(points)
    coords = get_point_list(points)

    for i in range(labels.max() + 1):
        points = coords[labels == i]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker=".",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
            zorder=2,
        )


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
