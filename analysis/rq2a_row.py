import json

MODELS = [
    ("resnet50", "ImageNet"),
    # ("mobilevit", "ImageNet")
]

MODEL_PATH = None
TARGET_LABEL = None
SEEDS = [0, 1, 2]

def generate_coverage_rows(json_path, batch_size, random_mutation=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    new_row = []

    for model_name, dataset_name in MODELS:
        lpipss = []
        ssimss = []
        class_coverages = []
        entropies = []
        stabilities = []
        num_classes = 0
        for seed in SEEDS:
            key = f"{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            key = f"yuan-{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            if random_mutation:
                key = key + "-rand"
            
            entry = data.get(key, {})
            num_classes =entry["number_of_classes"]
            lpipss.append(entry.get("naturalness", {}).get("mean_lpips", 0))
            ssimss.append(entry.get("naturalness", {}).get("mean_ssim", 0))
            class_coverages.append(entry.get("diversity", {}).get("class_covered", 0))
            entropies.append(entry.get("diversity", {}).get("scaled_entropy", 0))
            stabilities.append(entry.get("stability", {}).get("preserved_adversarial_rate", 0))

        lpips = sum(lpipss)/3
        ssim = sum(ssimss)/3
        cc = sum(class_coverages)/ 3
        e = sum(entropies)/3
        s = sum(stabilities)/3

        new_row.append(f"{lpips:.3f}")
        new_row.append(f"{ssim:.3f}")
        new_row.append(f"{s:.2f}")
        new_row.append(f"{cc*100/num_classes:.1f}")
        new_row.append(f"{e:.2f}")
        
    return new_row

if __name__ == "__main__":
    json_path = "results.json"   # path to your JSON file
    batch_size = 1              # set batch size here 

    new_row = generate_coverage_rows(
        json_path=json_path,
        batch_size=batch_size,
        random_mutation=False
    )

    print( " & ".join(new_row))
