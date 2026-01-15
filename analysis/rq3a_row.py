import json

MODELS = [
    ("resnet50", "UnsafeBench", "resnet50_unsafebench.pth"),
    ("mobilevit", "UnsafeBench", "mobilevit_unsafebench")
]

TARGET_LABEL = 0 # 0 safe, 1 unsafe
SEEDS = [0, 1, 2]

def generate_coverage_rows(json_path, batch_size, random_mutation=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    new_row = []

    for model_name, dataset_name, model_path in MODELS:
        lpipss = []
        faults = []
        new_cov = []
        for seed in SEEDS:
            key = f"{model_name}-{model_path}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            if random_mutation:
                key = key + "-rand"
            
            entry = data.get(key, {})
            faults.append(entry.get("generated_count", 0))
            lpipss.append(entry.get("naturalness", {}).get("mean_lpips", 0))
            cov = entry.get("coverage", {})
            new_cov.append(cov.get("new", 0))
            


        cov_avg = sum(new_cov) / 3
        new_row.append(f"{cov_avg:.2f}")

        fault_avg = sum(faults) / 3
        new_row.append(f"{fault_avg:.0f}")

        lpips = sum(lpipss)/3
        new_row.append(f"{lpips:.3f}")

    return new_row

if __name__ == "__main__":
    json_path = "results.json"   # path to your JSON file
    batch_size = 24              # set batch size here 

    new_row = generate_coverage_rows(
        json_path=json_path,
        batch_size=batch_size,
        random_mutation=False
    )

    print( " & ".join(new_row))
