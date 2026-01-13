import os
import math
import random
import numpy as np
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from PIL import Image
import lpips
import imagehash
from skimage.metrics import structural_similarity as ssim
import json
import torch

# -----------------------------
# Preprocessing
# -----------------------------
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -----------------------------
# Diversity Metrics
# -----------------------------

def lpips_distance(img1_path, img2_path):
    lpips_model = lpips.LPIPS(net='alex')
    img1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0)
    img2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        dist = lpips_model(img1, img2)
    return dist.item()

def ssim_distance(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert("L").resize((224, 224)))
    img2 = np.array(Image.open(img2_path).convert("L").resize((224, 224)))
    score = ssim(img1, img2, data_range=img2.max() - img2.min())
    return 1 - score  # Higher means more different

def hash_distance(img1_path, img2_path):
    h1 = imagehash.phash(Image.open(img1_path).convert("RGB").resize((224, 224)))
    h2 = imagehash.phash(Image.open(img2_path).convert("RGB").resize((224, 224)))
    return h1 - h2  # Hamming distance

def intra_class_metric(image_paths, labels, metric_func, max_pairs_per_class=1000, metric_name=""):
    from collections import defaultdict
    class_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        class_to_paths[label].append(path)

    class_metric = {}
    for c, paths in class_to_paths.items():
        n = len(paths)
        if n < 2:
            class_metric[c] = 0.0
            continue
        pairs = list(combinations(range(n), 2))
        if len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        scores = []
        for i, j in tqdm(pairs, desc=f"{metric_name} class {c}"):
            scores.append(metric_func(paths[i], paths[j]))
        class_metric[c] = np.mean(scores) if scores else 0.0
    # Aggregate
    mean_val = np.mean(list(class_metric.values()))
    min_val = np.min(list(class_metric.values()))
    max_val = np.max(list(class_metric.values()))
    return class_metric, mean_val, min_val, max_val

def scaled_entropy(pred_labels, num_classes):
    counts = Counter(pred_labels)
    total = len(pred_labels)
    probs = np.array([counts[c] / total for c in counts])
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    scaled = entropy / math.log(num_classes)
    return scaled

# -----------------------------
# Main Evaluation
# -----------------------------
import os
from collections import defaultdict

def evaluate_diversity(image_paths, num_classes=1000):
    labels = []
    for f in image_paths:
        parts = os.path.basename(f).split('_')
        adversarial_label = int(parts[4].split(".")[0])
        labels.append(adversarial_label)

    # Scaled Entropy (Label Diversity)
    label_div = scaled_entropy(labels, num_classes=num_classes)

    return {
        "count": len(labels),
        "class_covered": len(set(labels)),
        "scaled_entropy": round(label_div, 4),
    }

def calculate_diversity(aes_root, num_classes=1000):
    # Group by iteration
    # iter_groups = defaultdict(list)
    all_images = []

    for root, _, files in os.walk(aes_root):
        for f in files:

            path = os.path.join(root, f)

            # iter_groups[iteration].append(path)
            all_images.append(path)

    # Global result
    global_result = evaluate_diversity(all_images, num_classes=num_classes)

    return global_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--aes-root", type=str, required=True, help="Root directory of adversarial examples")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes in the dataset")
    args = parser.parse_args()

    results = calculate_diversity(args.aes_root, num_classes=args.num_classes)
    print(json.dumps(results, indent=4))