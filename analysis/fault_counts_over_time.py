import json
import numpy as np
import matplotlib.pyplot as plt
import os

outdir = "analysis"
os.makedirs(outdir, exist_ok=True)

# -------------------------------------------------
# Define fuzzers and their coverage.json paths
# -------------------------------------------------
FUZZERS = {
    "TF (N=1)": [
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/1/0/coverage.json",
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/1/1/coverage.json",
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/1/2/coverage.json",
    ],
    "TF-NoGrad (N=24)": [
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/0-rand/coverage.json",
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/1-rand/coverage.json",
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/2-rand/coverage.json",
    ],
    "TF (N=24)": [
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/0/coverage.json",
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/1/coverage.json",
        "adversarial-examples/resnet50/None/ImageNet/NLC/None/24/2/coverage.json",
    ],
    "Yuan et al.": [
        "adversarial-examples/yuan/ImageNet-resnet50-NLC-0/image/coverage.json",
        "adversarial-examples/yuan/ImageNet-resnet50-NLC-1/image/coverage.json",
        "adversarial-examples/yuan/ImageNet-resnet50-NLC-2/image/coverage.json",
    ],
}

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(6.5, 4.5))

for label, files in FUZZERS.items():
    all_times = []
    all_counts = []

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        time = np.array(data["time"])  / 60.0  # minutes
        counts = np.array(data["overall_ae_counts"])

        all_times.append(time)
        all_counts.append(counts)

    # Align runs by iteration index
    min_len = min(len(c) for c in all_counts)
    times = np.stack([t[:min_len] for t in all_times])
    counts = np.stack([c[:min_len] for c in all_counts])

    # Average across runs
    mean_time = times.mean(axis=0)
    mean_counts = counts.mean(axis=0)

    plt.plot(mean_time, mean_counts, lw=2, label=label)

# -------------------------------------------------
# Styling
# -------------------------------------------------
plt.xlabel("Time (minutes)", fontsize=13, fontweight="bold")
plt.ylabel("# Faults (Cumulative)", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, frameon=False)

plt.tight_layout()
plt.savefig(f"{outdir}/fault_counts_over_time.pdf", dpi=600, bbox_inches="tight")
plt.savefig(f"{outdir}/fault_counts_over_time.png", dpi=300, bbox_inches="tight")
