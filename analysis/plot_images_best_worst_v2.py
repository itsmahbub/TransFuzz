import os
import json
import torch
import lpips
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from matplotlib.gridspec import GridSpec

# =========================
# Configuration
# =========================
K = 3                        # number of examples per panel
IMG_SIZE = (224, 224)

TF_ROOT = "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/0"
YUAN_ROOT = "adversarial-examples/yuan/ImageNet-resnet50-NLC-0/image"

OUT_DIR = "analysis"
OUT_NAME = "naturalness-qualitative"

# =========================
# Load ImageNet label map
# =========================
with open("analysis/imagenet_label_map.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v.split(",")[0].strip() for k, v in label_map.items()}

# =========================
# LPIPS + transforms
# =========================
loss_fn = lpips.LPIPS(net="alex").eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# Metric computation
# =========================
def compute_metrics(orig_img, ae_img):
    t1 = transform(orig_img).unsqueeze(0)
    t2 = transform(ae_img).unsqueeze(0)
    with torch.no_grad():
        lp = loss_fn(t1, t2).item()
    ss = ssim(
        np.array(orig_img),
        np.array(ae_img),
        channel_axis=-1,
        data_range=255
    )
    return lp, ss

# =========================
# File matching utilities
# =========================
def collect_file_map(root_dir):
    file_map = {}
    for gt in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, gt)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            key = "_".join(fname.split("_")[:2])  # id_seq
            file_map[key] = (os.path.join(class_dir, fname), int(gt))
    return file_map

def collect_scored_examples(root_dir):
    orig_dir = os.path.join(root_dir, "orig")
    ae_dir   = os.path.join(root_dir, "aes")

    orig_map = collect_file_map(orig_dir)
    ae_map   = collect_file_map(ae_dir)

    shared_keys = sorted(set(orig_map) & set(ae_map))
    results = []

    for key in shared_keys:
        orig_path, gt_label = orig_map[key]
        ae_path, _ = ae_map[key]

        orig_img = Image.open(orig_path).convert("RGB").resize(IMG_SIZE)
        ae_img   = Image.open(ae_path).convert("RGB").resize(IMG_SIZE)

        lp, ss = compute_metrics(orig_img, ae_img)

        # adversarial label = 4th underscore-separated field
        fname = os.path.basename(ae_path)
        parts = fname.split("_")
        try:
            adv_label = int(parts[4].split(".")[0])
        except Exception:
            adv_label = None

        results.append({
            "key": key,
            "orig": orig_img,
            "orig_path": orig_path,
            "ae": ae_img,
            "ae_path": ae_path,
            "gt": gt_label,
            "adv": adv_label,
            "lpips": lp,
            "ssim": ss
        })

    return sorted(results, key=lambda x: x["lpips"])

# =========================
# Select distinct GT labels
# =========================
def select_unique_gt(scored, k, reverse=False):
    used = set()
    selected = []
    seq = reversed(scored) if reverse else scored
    for ex in seq:
        if ex["gt"] not in used:
            selected.append(ex)
            used.add(ex["gt"])
        if len(selected) == k:
            break
    return selected

# =========================
# Plotting utilities
# =========================
def plot_panel(fig, gs, title, examples, show_row_labels):
    print(examples)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.2, 0.5, title,
        fontsize=12, fontweight="bold",
        ha="left", va="center"
    )

    for i, ex in enumerate(examples):
        ax_o = fig.add_subplot(gs[1, i])
        ax_a = fig.add_subplot(gs[2, i])

        if i==0 and show_row_labels:
            ax_o.text(
                -30,110, "Original",
                rotation=90,
                fontsize=8,
                fontweight="semibold",
                va="center",
                ha="center"
            )
            ax_a.text(
                -30, 110, "Fault",
                rotation=90,
                fontsize=8,
                fontweight="semibold",
                va="center",
                ha="center"
            )
        
        ax_o.imshow(ex["orig"])
        ax_o.axis("off")

        gt_name  = label_map.get(ex["gt"], str(ex["gt"]))

        ax_o.set_title(
            gt_name.title(),
            fontsize=10, pad=1
        )

        ax_a.imshow(ex["ae"])
        ax_a.axis("off")

        
        adv_name = label_map.get(ex["adv"], str(ex["adv"]))


        ax_a.text(
            0.5, -0.01,   # x=centered, y=below image
            f"{adv_name.title()}\n"
            f"LPIPS={ex['lpips']:.3f}\nSSIM={ex['ssim']:.3f}",
            fontsize=10,
            ha="center",
            va="top",
            transform=ax_a.transAxes
        )


# =========================
# Main
# =========================
def main():
    tf = collect_scored_examples(TF_ROOT)
    yu = collect_scored_examples(YUAN_ROOT)

    panels = [
        ("(a) TransFuzz - Most natural",  select_unique_gt(tf, K, False)),
        ("(b) TransFuzz - Least natural", select_unique_gt(tf, K, True)),
        ("(c) Yuan et al. - Most natural",  select_unique_gt(yu, K, False)),
        ("(d) Yuan et al. - Least natural", select_unique_gt(yu, K, True)),
    ]

    fig = plt.figure(figsize=(3.4*K, 7.5))  # wider, shorter
    outer = GridSpec(
        2, 2,
        figure=fig,
        wspace=0.025,   # space BETWEEN panels
        hspace=0.22
    )

    panel_positions = [
        (0, 0),  # (a)
        (0, 1),  # (b)
        (1, 0),  # (c)
        (1, 1),  # (d)
    ]

    for (title, ex), (r, c) in zip(panels, panel_positions):
        inner = outer[r, c].subgridspec(
            3, K,
            height_ratios=[0.4, 1, 1],
            hspace=0.1,
            wspace=-0.35 
        )
     

        plot_panel(fig, inner, title, ex, show_row_labels=(c == 0))

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(f"{OUT_DIR}/{OUT_NAME}.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/{OUT_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {OUT_DIR}/{OUT_NAME}.pdf")

# =========================
# Run
# =========================
if __name__ == "__main__":
    main()
